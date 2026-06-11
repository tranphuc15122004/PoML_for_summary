"""
SFT Batch Size Smoke Test
Mục đích: Verify production config (batch=4, grad_accum=4, seq=3072, bf16)
chạy được trên H200 mà không OOM.
"""
import os, sys, json, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_DATASETS_CACHE'] = '/scratch/jp09/dd9648/.cache/huggingface/datasets'

import torch
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)

LOCAL_MODEL = '/g/data/hn98/dd9648/models/Qwen2.5-3B-Instruct'

log.info(f"GPU: {torch.cuda.get_device_name(0)}")
log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── 1. Tạo synthetic data với seq dài (3072 tokens worth) ──────────────────
from datasets import Dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(LOCAL_MODEL, local_files_only=True)
log.info("Tokenizer loaded")

# Tạo samples đủ dài để test truncation/padding ở seq_len=3072
long_text = "Đây là nội dung bài báo rất dài để kiểm tra khả năng xử lý của mô hình. " * 60

samples = []
for i in range(16):  # 16 samples → 4 batches × batch_size=4
    samples.append({
        'messages': [
            {'role': 'system', 'content': 'Bạn là chuyên gia tóm tắt văn bản tiếng Việt.'},
            {'role': 'user', 'content': f'Tóm tắt (khoảng 80 từ, phong cách báo chí):\n\n{long_text}'},
            {'role': 'assistant', 'content': 'Bài báo trình bày nội dung quan trọng về vấn đề kinh tế xã hội tại Việt Nam trong giai đoạn phát triển mạnh mẽ.'}
        ],
        'meta': {'length_requirement': 'khoảng 80 từ', 'style': 'báo chí'}
    })

raw_ds = Dataset.from_list(samples)

# Tokenize như train_sft.py
def tokenize(ex):
    text = tok.apply_chat_template(ex['messages'], tokenize=False, add_generation_prompt=False)
    enc = tok(text, truncation=True, max_length=3072)
    enc['labels'] = enc['input_ids'].copy()
    return enc

all_cols = raw_ds.column_names
ds = raw_ds.map(tokenize, remove_columns=all_cols)
log.info(f"Dataset: {len(ds)} samples")
log.info(f"Token lengths: min={min(len(x) for x in ds['input_ids'])}, max={max(len(x) for x in ds['input_ids'])}")

# ── 2. Load model với production LoRA config ───────────────────────────────
from SFT_GRPO.config import SFTConfig, ModelConfig
from SFT_GRPO.train_sft import load_model, load_tokenizer, get_lora_config
from trl import SFTTrainer, SFTConfig as TRLSFTConfig

cfg = SFTConfig(
    model=ModelConfig(model_name_or_path=LOCAL_MODEL),
    per_device_train_batch_size=4,      # ← PRODUCTION batch size
    gradient_accumulation_steps=4,      # ← effective batch = 16
    max_seq_length=3072,                # ← PRODUCTION seq length
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,        # ← production default
    logging_steps=1,
    save_steps=9999,
    eval_strategy='no',
    report_to='none',
    output_dir='/scratch/jp09/dd9648/PoML_for_summary/models/sft_batch_test',
)

log.info("Loading model...")
model = load_model(cfg.model)
lora_cfg = get_lora_config(cfg.model)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
log.info(f"Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")

# ── 3. Run 3 training steps ────────────────────────────────────────────────
training_args = TRLSFTConfig(
    output_dir=cfg.output_dir,
    per_device_train_batch_size=cfg.per_device_train_batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    learning_rate=5e-5,
    max_seq_length=cfg.max_seq_length,
    bf16=cfg.bf16,
    fp16=cfg.fp16,
    gradient_checkpointing=cfg.gradient_checkpointing,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    logging_steps=1,
    save_steps=9999,
    eval_strategy='no',
    max_steps=3,          # chỉ 3 steps để test throughput
    report_to='none',
    dataset_text_field=None,
    packing=False,
    ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    processing_class=tok,
    peft_config=lora_cfg,
)

log.info("=" * 60)
log.info("Starting production-config SFT (3 steps)...")
log.info(f"  batch_size={cfg.per_device_train_batch_size}, grad_accum={cfg.gradient_accumulation_steps}")
log.info(f"  effective_batch={cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
log.info(f"  max_seq_length={cfg.max_seq_length}, bf16={cfg.bf16}")
log.info(f"  gradient_checkpointing={cfg.gradient_checkpointing}")
log.info("=" * 60)

import time
t0 = time.time()
trainer.train()
elapsed = time.time() - t0

# Log LoRA params AFTER SFTTrainer applies LoRA
trained_model = trainer.model
trainable_after = sum(p.numel() for p in trained_model.parameters() if p.requires_grad)
total_after = sum(p.numel() for p in trained_model.parameters())
log.info(f"Trainable params (after LoRA): {trainable_after/1e6:.1f}M / {total_after/1e6:.1f}M ({100*trainable_after/total_after:.2f}%)")

# Peak VRAM
peak_mb = torch.cuda.max_memory_allocated() / 1e9
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9

print()
print("=" * 60)
print("SFT BATCH SIZE TEST RESULTS")
print("=" * 60)
print(f"  Status       : PASS")
print(f"  Batch size   : {cfg.per_device_train_batch_size} (effective: {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps})")
print(f"  Max seq len  : {cfg.max_seq_length}")
print(f"  bf16         : {cfg.bf16}")
print(f"  Peak VRAM    : {peak_mb:.1f} GB / {total_vram:.1f} GB ({100*peak_mb/total_vram:.1f}% used)")
print(f"  Wall time    : {elapsed:.1f}s for 3 steps")
print(f"  Throughput   : {3/elapsed*60:.1f} steps/min")
print("=" * 60)
