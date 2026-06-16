#!/usr/bin/env python
"""Direct smoke test for GRPO pipeline (Tesla T4 16GB compatible)."""

import sys, os, json, gc, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("smoke_grpo")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from SFT_GRPO.rewards import compute_all_rewards, accuracy_reward, length_reward
from SFT_GRPO.config import GRPOConfig, ModelConfig

PASS = 0
FAIL = 0
RESULTS = []

def test(name, fn):
    global PASS, FAIL
    logger.info(f"\n=== TEST: {name} ===")
    try:
        fn()
        logger.info(f"  ✅ PASS: {name}")
        PASS += 1
        RESULTS.append((name, "PASS", ""))
    except Exception as e:
        logger.error(f"  ❌ FAIL: {name}: {e}")
        import traceback
        traceback.print_exc()
        FAIL += 1
        RESULTS.append((name, "FAIL", str(e)))

# ============================================================
# 1. REWARD FUNCTIONS
# ============================================================
def test_rewards():
    # Length reward - exact
    gen10 = ' '.join(['word'] * 10)
    r = length_reward(gen10, 'khoảng 10 từ')
    assert r == 1.0, f"khoảng 10 từ, actual=10 -> {r}"
    
    # Within tolerance (9 words, ±20% of 10 = 2, so 8-12)
    gen9 = ' '.join(['word'] * 9)
    r = length_reward(gen9, 'khoảng 10 từ')
    assert r == 1.0, f"khoảng 10 từ, actual=9 -> {r}"
    
    # Far outside
    gen20 = ' '.join(['word'] * 20)
    r = length_reward(gen20, 'khoảng 10 từ')
    assert r == 0.0, f"khoảng 10 từ, actual=20 -> {r}"
    
    # Range
    gen25 = ' '.join(['word'] * 25)
    r = length_reward(gen25, 'trong khoảng 20-30 từ')
    assert r == 1.0, f"range 20-30, actual=25 -> {r}"
    
    # Max
    gen50 = ' '.join(['word'] * 50)
    r = length_reward(gen50, 'không quá 50 từ')
    assert r == 1.0, f"max 50, actual=50 -> {r}"
    
    # Accuracy
    r = accuracy_reward('trời đẹp', 'trời đẹp')
    assert r == 1.0, f"exact match -> {r}"
    r = accuracy_reward('', 'trời đẹp')
    assert r == 0.0, f"empty gen -> {r}"
    
    # Composite (with sentence requirement)
    r = compute_all_rewards(
        generated='Hà Nội thông qua nghị quyết phát triển kinh tế số.',
        reference='Hà Nội thông qua nghị quyết kinh tế số.',
        length_requirement='khoảng 10 từ',
        sentence_requirement='khoảng 1 câu',
    )
    assert 0 <= r['total'] <= 1
    assert 'sentence' in r
    logger.info(f"  Composite reward: {r}")

    # Composite (no sentence requirement)
    r2 = compute_all_rewards(
        generated='Hà Nội thông qua nghị quyết phát triển kinh tế số.',
        reference='Hà Nội thông qua nghị quyết kinh tế số.',
        length_requirement='khoảng 10 từ',
    )
    assert 0 <= r2['total'] <= 1
    assert 'sentence' not in r2
    logger.info(f"  Composite reward (no sent): {r2}")

# ============================================================
# 2. MODEL LOADING (4-bit)
# ============================================================
def test_model_loading():
    logger.info("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"  Tokenizer OK. pad={tokenizer.pad_token_id}")
    
    logger.info("  Loading 4-bit model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    mem = torch.cuda.memory_allocated() / 1e9
    logger.info(f"  Model loaded: {model.num_parameters() / 1e9:.2f}B params, VRAM: {mem:.2f} GB")
    
    # Add LoRA
    lora_config = LoraConfig(
        r=8,  # smaller rank for smoke test
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    mem2 = torch.cuda.memory_allocated() / 1e9
    logger.info(f"  LoRA applied. VRAM: {mem2:.2f} GB")
    
    # Test forward pass
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Tóm tắt (khoảng 10 từ):\n\nHà Nội phát triển kinh tế số."}],
        tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=20, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    logger.info(f"  Generation test: '{generated[:100]}'")
    mem3 = torch.cuda.memory_allocated() / 1e9
    logger.info(f"  After generation VRAM: {mem3:.2f} GB")
    
    # Clean up
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"  Cleanup OK, VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ============================================================
# 3. GRPO TRAINING STEP (standalone test)
# ============================================================
def test_grpo_step():
    logger.info("  Setting up models for GRPO step...")
    
    # Load reference model (4-bit, frozen)
    logger.info("  Loading reference model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    logger.info(f"  Reference model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Load policy model (4-bit + LoRA, trainable)
    logger.info("  Loading policy model...")
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    base.config.use_cache = False
    lora_config = LoraConfig(
        r=8, lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    policy_model = get_peft_model(base, lora_config)
    policy_model.print_trainable_parameters()
    logger.info(f"  Policy model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create a tiny batch
    prompts = [
        [{"role": "user", "content": "Tóm tắt (khoảng 15 từ, phong cách báo chí):\n\nHội đồng Nhân dân thành phố Hà Nội vừa thông qua nghị quyết về phát triển kinh tế số giai đoạn 2024-2030 với tổng vốn đầu tư dự kiến lên đến 50.000 tỷ đồng."}],
        [{"role": "user", "content": "Tóm tắt (khoảng 10 từ, phong cách ngắn gọn):\n\nBộ Giáo dục và Đào tạo công bố kết quả thi tốt nghiệp THPT năm 2025 với tỷ lệ đỗ tốt nghiệp đạt 98.2% tăng nhẹ so với năm trước."}],
    ]
    references = [
        "Hà Nội thông qua nghị quyết phát triển kinh tế số 2024-2030 với 50.000 tỷ đồng.",
        "Kết quả thi THPT 2025: tỷ lệ đỗ tốt nghiệp đạt 98.2%.",
    ]
    meta_list = [
        {"length_requirement": "khoảng 15 từ", "sentence_requirement": "khoảng 1 câu"},
        {"length_requirement": "khoảng 10 từ", "sentence_requirement": "khoảng 1 câu"},
    ]
    
    # Format prompts
    prompt_texts = [
        tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in prompts
    ]
    
    # ROLLOUT: generate K=2 completions per prompt
    num_gen = 2
    expanded_prompts = [pt for pt in prompt_texts for _ in range(num_gen)]
    
    logger.info("  Running rollout (generation)...")
    prompt_encodings = tokenizer(
        expanded_prompts, padding=True, truncation=True,
        max_length=384, return_tensors="pt",
    ).to(policy_model.device)
    
    output_ids = policy_model.generate(
        **prompt_encodings,
        max_new_tokens=64,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    
    prompt_len = prompt_encodings.input_ids.shape[1]
    gen_ids = output_ids.sequences[:, prompt_len:]
    gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    
    logger.info(f"  Generated {len(gen_texts)} completions")
    for i, gt in enumerate(gen_texts):
        wc = len(gt.split())
        logger.info(f"    [{i}] ({wc} từ) {gt[:80]}...")
    
    # REWARD computation
    logger.info("  Computing rewards...")
    batch_size = len(prompts)
    rewards = torch.zeros(batch_size * num_gen)
    
    for i in range(batch_size):
        ref = references[i]
        meta = meta_list[i]
        for k in range(num_gen):
            idx = i * num_gen + k
            gen = gen_texts[idx] if idx < len(gen_texts) else ""
            rd = compute_all_rewards(
                generated=gen, reference=ref if ref else gen,
                length_requirement=meta.get("length_requirement", "khoảng 50 từ"),
                sentence_requirement=meta.get("sentence_requirement", None),
                w_acc=0.5, w_len=0.3, w_sent=0.2,
            )
            rewards[idx] = rd["total"]
            logger.info(f"    [{idx}] reward={rd['total']:.4f} (acc={rd['accuracy']:.4f}, len={rd['length']:.4f}, sent={rd.get('sentence', 'N/A')})")
    
    # ADVANTAGE: group normalization
    rewards_2d = rewards.view(batch_size, num_gen)
    mean_rewards = rewards_2d.mean(dim=-1, keepdim=True)
    std_rewards = rewards_2d.std(dim=-1, keepdim=True) + 1e-8
    advantages_2d = (rewards_2d - mean_rewards) / std_rewards
    advantages = advantages_2d.flatten()
    logger.info(f"  Advantages: {advantages.tolist()}")
    
    # Compute new logprobs (with gradients)
    logger.info("  Computing policy gradients...")
    full_ids = torch.cat([prompt_encodings.input_ids, gen_ids], dim=-1)
    attention_mask = torch.ones_like(full_ids)
    
    outputs = policy_model(input_ids=full_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    gen_logits = logits[:, prompt_len - 1: -1]
    log_probs = torch.log_softmax(gen_logits, dim=-1)
    
    # KL divergence with reference model
    with torch.no_grad():
        ref_outputs = ref_model(input_ids=full_ids, attention_mask=attention_mask)
        ref_logits = ref_outputs.logits
        ref_gen_logits = ref_logits[:, prompt_len - 1: -1]
        ref_log_probs = torch.log_softmax(ref_gen_logits, dim=-1)
    
    gen_tokens = gen_ids
    token_ref_logp = ref_log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)
    token_pol_logp = log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)
    r = (token_ref_logp - token_pol_logp).clamp(min=-20.0, max=20.0)
    kl_penalty = (torch.exp(r) - r - 1).mean()
    logger.info(f"  KL penalty: {kl_penalty.item():.4f}")
    
    # Policy gradient loss
    policy_loss = 0.0
    n_tokens = 0
    for i in range(len(gen_texts)):
        old_lp = torch.zeros(len(gen_ids[i]))  # simplified: uniform old logprobs
        new_lp = log_probs[i, torch.arange(len(gen_ids[i])), gen_ids[i]]
        rho = torch.exp(new_lp - old_lp.to(new_lp.device))
        adv = advantages[i]
        surr1 = rho * adv
        surr2 = torch.clamp(rho, 0.8, 1.2) * adv
        policy_loss += -torch.min(surr1, surr2).sum()
        n_tokens += len(gen_ids[i])
    
    policy_loss = policy_loss / max(n_tokens, 1)
    beta = 0.04
    total_loss = policy_loss + beta * kl_penalty
    logger.info(f"  Policy loss: {policy_loss.item():.4f}")
    logger.info(f"  Total loss: {total_loss.item():.4f}")
    
    # BACKWARD
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in policy_model.parameters() if p.requires_grad], max_norm=1.0
    )
    logger.info(f"  Grad norm: {grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm:.4f}")
    
    # Count trainable params with grads
    grad_params = sum(p.grad is not None and p.grad.abs().sum() > 0 for p in policy_model.parameters())
    logger.info(f"  Params with non-zero gradient: {grad_params}")
    
    # Clean up
    del policy_model, ref_model, tokenizer, base
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"  Cleanup OK. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ============================================================
# Run all tests
# ============================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("SMOKE TEST: GRPO Pipeline")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A")
    logger.info("=" * 60)
    
    test("Reward functions", test_rewards)
    test("Model loading (4-bit) + generation", test_model_loading)
    
    logger.info("\n" + "=" * 60)
    logger.info("ATTEMPTING GRPO STEP TEST (may OOM on 16GB)...")
    logger.info("=" * 60)
    
    try:
        test("GRPO training step", test_grpo_step)
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"  ❌ FAIL: GRPO step - CUDA OOM: {e}")
        logger.error("  This is expected on Tesla T4 16GB with dual 3B models.")
        logger.error("  GRPO needs ~24-30GB VRAM for Qwen2.5-3B policy + reference models.")
        FAIL += 1
        RESULTS.append(("GRPO training step", "FAIL (OOM - expected)", str(e)))
    except Exception as e:
        logger.error(f"  ❌ FAIL: GRPO step - {e}")
        import traceback
        traceback.print_exc()
        FAIL += 1
        RESULTS.append(("GRPO training step", "FAIL", str(e)))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SMOKE TEST RESULTS")
    logger.info("=" * 60)
    for name, status, msg in RESULTS:
        icon = "✅" if status == "PASS" else "❌"
        logger.info(f"  {icon} {name}: {status}{' - ' + msg if msg else ''}")
    logger.info(f"\n  PASSED: {PASS} / {PASS + FAIL}")
    logger.info(f"  FAILED: {FAIL} / {PASS + FAIL}")
    logger.info("=" * 60)
    
    if FAIL > 0:
        logger.warning("Some tests failed. See details above.")
        sys.exit(1)
    else:
        logger.info("ALL TESTS PASSED!")
        sys.exit(0)
