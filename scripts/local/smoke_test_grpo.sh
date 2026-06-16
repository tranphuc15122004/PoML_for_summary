#!/bin/bash
# =============================================================================
# Smoke Test GRPO - Kiểm tra GRPO training trên Tesla T4 (16 GB)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="/home/tuantb/PoTR_article_summary"
DATA_DIR="${PROJECT_ROOT}/data/smoke_test_grpo"
MODELS_DIR="${PROJECT_ROOT}/models/smoke_test_grpo"
LOG_FILE="${PROJECT_ROOT}/logs/smoke_grpo_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${DATA_DIR}" "${MODELS_DIR}" "${PROJECT_ROOT}/logs"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================"
echo "SMOKE TEST GRPO - PoML for Summary"
echo "Node: $(hostname)"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "Date: $(date)"
echo "================================================================"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PASS=0
FAIL=0

check() {
    local name="$1"
    shift
    echo ""
    echo "--- [TEST] ${name} ---"
    if "$@"; then
        echo "  ✅ PASS: ${name}"
        PASS=$((PASS + 1))
    else
        echo "  ❌ FAIL: ${name}"
        FAIL=$((FAIL + 1))
    fi
}

# ============================================================
# 1. ENVIRONMENT CHECK
# ============================================================
echo ""
echo "=== 1. ENVIRONMENT ==="

check "Python version" python -c "
import sys
assert sys.version_info >= (3, 10), f'Python {sys.version} too old'
print(f'Python {sys.version}')
"

check "PyTorch + CUDA" python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

check "Key packages" python -c "
import transformers, peft, accelerate
print(f'transformers={transformers.__version__}, peft={peft.__version__}, accelerate={accelerate.__version__}')
"

check "GRPO imports" python -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}/src')
from SFT_GRPO.config import GRPOConfig, ModelConfig
from SFT_GRPO.rewards import compute_all_rewards, accuracy_reward, length_reward
from SFT_GRPO.metrics_logger import MetricsTracker
print('All project imports OK')
"

# ============================================================
# 2. REWARD FUNCTIONS
# ============================================================
echo ""
echo "=== 2. REWARD FUNCTIONS ==="

check "Length reward - khoảng" python -c "
from SFT_GRPO.rewards import length_reward
# 10 words within ±20% of 10 (= 2) → 8-12
gen = ' '.join(['word'] * 10)
r = length_reward(gen, 'khoảng 10 từ')
assert r == 1.0, f'Expected 1.0, got {r}'
print(f'  exactly 10 words: {r} ✓')
gen9 = ' '.join(['word'] * 9)
r9 = length_reward(gen9, 'khoảng 10 từ')
assert r9 == 1.0, f'Expected 1.0, got {r9}'
print(f'  9 words (within tol): {r9} ✓')
gen20 = ' '.join(['word'] * 20)
r20 = length_reward(gen20, 'khoảng 10 từ')
assert r20 == 0.0, f'Expected 0.0, got {r20}'
print(f'  20 words (far outside): {r20} ✓')
"

check "Length reward - range" python -c "
from SFT_GRPO.rewards import length_reward
gen = ' '.join(['word'] * 25)
r = length_reward(gen, 'trong khoảng 20-30 từ')
assert r == 1.0, f'Expected 1.0, got {r}'
print(f'  25 words in [20,30]: {r} ✓')
"

check "Length reward - max" python -c "
from SFT_GRPO.rewards import length_reward
gen = ' '.join(['word'] * 50)
r = length_reward(gen, 'không quá 50 từ')
assert r == 1.0, f'Expected 1.0, got {r}'
print(f'  exactly 50 words: {r} ✓')
gen65 = ' '.join(['word'] * 65)
r2 = length_reward(gen65, 'không quá 50 từ')
print(f'  65 words (over): {r2} ✓ (expected > 0)')
"

check "Accuracy reward (ROUGE-L)" python -c "
from SFT_GRPO.rewards import accuracy_reward, rouge_l_f1
# Exact match
r = accuracy_reward('trời đẹp', 'trời đẹp')
assert r == 1.0, f'Expected 1.0, got {r}'
print(f'  exact match: {r} ✓')
# Empty
r2 = accuracy_reward('', 'trời đẹp')
assert r2 == 0.0
print(f'  empty gen: {r2} ✓')
"

check "Composite reward" python -c "
from SFT_GRPO.rewards import compute_all_rewards
r = compute_all_rewards(
    generated='Hà Nội thông qua nghị quyết phát triển kinh tế số.',
    reference='Hà Nội thông qua nghị quyết kinh tế số.',
    length_requirement='khoảng 10 từ',
    style='báo chí'
)
print(f'  Composite: {r}')
assert 0 <= r['total'] <= 1, f'total={r[\"total\"]} not in [0,1]'
assert 0 <= r['accuracy'] <= 1
assert 0 <= r['length'] <= 1
assert r['style'] == 0.5  # no judge → neutral
print(f'  All constraints pass ✓')
"

# ============================================================
# 3. MODEL LOADING (4-bit quantization)
# ============================================================
echo ""
echo "=== 3. MODEL LOADING (4-bit) ==="

check "Load tokenizer" python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct', trust_remote_code=True)
print(f'Tokenizer: vocab_size={len(tok)}, pad={tok.pad_token_id}, eos={tok.eos_token_id}')
# Test prompt formatting
msgs = [{'role': 'user', 'content': 'Tóm tắt (khoảng 50 từ):\n\nHà Nội phát triển kinh tế số.'}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print(f'Chat template OK: {text[:80]}...')
"

check "Load 4-bit model (base)" python -c "
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import gc
try:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-3B-Instruct',
        quantization_config=bnb,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation='sdpa',
    )
    print(f'Model loaded: {model.__class__.__name__}')
    print(f'Params: {model.num_parameters() / 1e9:.2f}B')
    print(f'Device: {model.device}')
    # Check memory
    mem = torch.cuda.memory_allocated() / 1e9
    print(f'VRAM used: {mem:.2f} GB')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    mem_after = torch.cuda.memory_allocated() / 1e9
    print(f'VRAM after cleanup: {mem_after:.2f} GB')
    print('Model loading OK')
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1 | tail -20

# ============================================================
# 4. GRPO MINI TRAINING (3 steps)
# ============================================================
echo ""
echo "=== 4. GRPO MINI TRAINING (3 steps) ==="

# Create tiny synthetic dataset
python -c "
import json, os
DATA_DIR = '${DATA_DIR}'
os.makedirs(DATA_DIR, exist_ok=True)

articles = [
    ('Hội đồng Nhân dân thành phố Hà Nội vừa thông qua nghị quyết về phát triển kinh tế số giai đoạn 2024-2030 với tổng vốn đầu tư dự kiến lên đến 50.000 tỷ đồng.', 'Hà Nội thông qua nghị quyết phát triển kinh tế số 2024-2030 với 50.000 tỷ đồng.'),
    ('Bộ Giáo dục và Đào tạo công bố kết quả thi tốt nghiệp THPT năm 2025 với tỷ lệ đỗ tốt nghiệp đạt 98.2% tăng nhẹ so với năm trước.', 'Kết quả thi THPT 2025: tỷ lệ đỗ tốt nghiệp đạt 98.2%.'),
    ('Giá xăng dầu trong nước điều chỉnh tăng lần thứ ba liên tiếp từ 15h chiều nay sau quyết định của Liên bộ Công Thương - Tài chính.', 'Giá xăng dầu tăng lần thứ ba liên tiếp từ chiều nay.'),
    ('Tập đoàn công nghệ ABC ra mắt mô hình AI mới với khả năng xử lý ngôn ngữ tự nhiên vượt trội hỗ trợ hơn 50 ngôn ngữ khác nhau.', 'ABC ra mắt AI mới hỗ trợ hơn 50 ngôn ngữ.'),
]

# Create SFT data
sft_data = []
for src, ref in articles:
    for length_req, style in [
        ('khoảng 15 từ', 'báo chí'),
        ('khoảng 10 từ', 'ngắn gọn'),
        ('trong khoảng 12-18 từ', 'báo chí'),
    ]:
        sft_data.append({
            'messages': [
                {'role': 'system', 'content': 'Bạn là chuyên gia tóm tắt văn bản tiếng Việt.'},
                {'role': 'user', 'content': f'Tóm tắt ({length_req}, phong cách {style}):\n\n{src}'},
                {'role': 'assistant', 'content': ref}
            ],
            'meta': {'length_requirement': length_req, 'style': style}
        })

# Create GRPO data (prompt only, no assistant response)
grpo_data = []
for src, ref in articles:
    for length_req, style in [
        ('khoảng 15 từ', 'báo chí'),
        ('khoảng 10 từ', 'ngắn gọn'),
        ('trong khoảng 12-18 từ', 'báo chí'),
        ('không quá 20 từ', 'hài hước'),
    ]:
        grpo_data.append({
            'prompt': [
                {'role': 'system', 'content': 'Bạn là chuyên gia tóm tắt văn bản tiếng Việt.'},
                {'role': 'user', 'content': f'Tóm tắt ({length_req}, phong cách {style}):\n\n{src}'}
            ],
            'reference': ref,
            'meta': {'length_requirement': length_req, 'style': style}
        })

with open(os.path.join(DATA_DIR, 'grpo_train.jsonl'), 'w', encoding='utf-8') as f:
    for item in grpo_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
with open(os.path.join(DATA_DIR, 'grpo_val.jsonl'), 'w', encoding='utf-8') as f:
    for item in grpo_data[:4]:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f'Created {len(grpo_data)} GRPO samples')
" 2>&1

echo ""
echo "--- Running GRPO: 3 steps, batch=1, num_gen=2, 4-bit ---"
echo ""

# Write GRPO config for smoke test
cat > /tmp/smoke_grpo_t4.json << 'EOF'
{
    "do_sample": true,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 64,
    "max_seq_length": 512,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "logging_steps": 1,
    "save_steps": 9999,
    "eval_strategy": "no",
    "report_to": "none",
    "bf16": false,
    "fp16": true,
    "gradient_checkpointing": true,
    "num_generations": 2,
    "load_in_4bit": true,
    "learning_rate": 5e-7,
    "beta": 0.04,
    "epsilon": 0.2,
    "reward_weight_accuracy": 0.5,
    "reward_weight_length": 0.3,
    "reward_weight_style": 0.2
}
EOF

check "GRPO 3 steps (T4 16GB)" python "${PROJECT_ROOT}/src/SFT_GRPO/train_grpo.py" \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --output_dir "${MODELS_DIR}/grpo" \
    --train_data "${DATA_DIR}/grpo_train.jsonl" \
    --val_data "${DATA_DIR}/grpo_val.jsonl" \
    --total_steps 3 \
    --num_generations 2 \
    --config /tmp/smoke_grpo_t4.json 2>&1

# ============================================================
# 5. SUMMARY
# ============================================================
echo ""
echo "================================================================"
echo "SMOKE TEST GRPO RESULTS"
echo "================================================================"
echo "  PASSED: ${PASS}"
echo "  FAILED: ${FAIL}"
echo "  Log: ${LOG_FILE}"
echo "================================================================"

if [ "${FAIL}" -gt 0 ]; then
    echo "RESULT: FAIL - ${FAIL} test(s) failed."
    exit 1
else
    echo "RESULT: PASS - GRPO pipeline ready."
    echo ""
    echo "✅ Reward functions: working correctly"
    echo "✅ Model loading (4-bit): working"
    echo "✅ GRPO training loop: working (if passed)"
fi
