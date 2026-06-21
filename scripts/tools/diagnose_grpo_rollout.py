#!/usr/bin/env python
"""Diagnose why GRPO rollouts produce garbage (R_acc ≈ 0).

Loads the raw base model, the SFT checkpoint, and a GRPO checkpoint, then
generates on a few val prompts under three regimes to isolate the cause:

  A. eval-clean    : model.eval(), use_cache=True, greedy, NO penalties
                     → what evaluate.py does. Shows the model's "true" quality.
  B. rollout-replica: model.train() (LoRA dropout ON), use_cache=False,
                     do_sample=True T=0.7 top_p=0.9, repetition_penalty=1.3,
                     no_repeat_ngram_size=3 → exactly what train_grpo.py rollout does.
  C. rollout-eval  : model.eval(), use_cache=True, same sampling + penalties as B
                     → isolates the train-mode/dropout effect from the penalties.

If A is clean but B is garbage → the rollout config is the culprit.
If A is garbage → the model/prompt format itself is broken.
B vs C separates train-mode dropout from the decoding penalties.
"""

from __future__ import annotations

import json
import os
import sys
import types

# bitsandbytes references triton.ops which was removed in triton 2.x. PEFT imports
# bitsandbytes during PeftModel.from_pretrained(); stub the missing submodule so
# the import succeeds (same shim train_grpo.py applies at startup).
if "triton.ops" not in sys.modules:
    _triton_ops = types.ModuleType("triton.ops")
    _triton_perf = types.ModuleType("triton.ops.matmul_perf_model")
    _triton_perf.early_config_prune = lambda *a, **kw: None
    _triton_perf.estimate_matmul_time = lambda *a, **kw: 0.0
    sys.modules["triton.ops"] = _triton_ops
    sys.modules["triton.ops.matmul_perf_model"] = _triton_perf

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = "/scratch/jp09/dd9648/PoML_for_summary"
sys.path.insert(0, os.path.join(PROJECT, "src"))
from SFT_GRPO.rewards import compute_all_rewards  # noqa: E402

BASE_INSTRUCT = "/g/data/hn98/dd9648/models/Qwen3-4B"
SFT_CKPT = f"{PROJECT}/models/sft_qwen3_4b_instruct/final"
GRPO_CKPT = f"{PROJECT}/models/grpo_qwen3_4b_instruct_sft_v3/checkpoint-400"
VAL_DATA = f"{PROJECT}/data/grpo_val.jsonl"
N_SAMPLES = 3


def fmt_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def load(model_path, adapter=None):
    """Load base (+ optional UNMERGED LoRA adapter so dropout layers exist)."""
    tok = AutoTokenizer.from_pretrained(BASE_INSTRUCT, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)  # NOT merged → dropout present
    return model, tok


def gen(model, tok, prompt_text, *, train_mode, use_cache, do_sample, penalties):
    if train_mode:
        model.train()
    else:
        model.eval()
    model.config.use_cache = use_cache
    inputs = tok(prompt_text, return_tensors="pt", truncation=True, max_length=3072 - 80).to(model.device)
    kwargs = dict(
        max_new_tokens=80,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if do_sample:
        kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)
    else:
        kwargs.update(do_sample=False)
    if penalties:
        kwargs.update(repetition_penalty=1.3, no_repeat_ngram_size=3)
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    plen = inputs.input_ids.shape[1]
    return tok.decode(out[0][plen:], skip_special_tokens=True).strip()


def main():
    samples = []
    with open(VAL_DATA) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= N_SAMPLES:
                break

    configs = [
        ("base_instruct (no adapter)", BASE_INSTRUCT, None),
        ("SFT/final", BASE_INSTRUCT, SFT_CKPT),
        ("GRPO sft_v3/checkpoint-400", BASE_INSTRUCT, GRPO_CKPT),
    ]

    # Dump the exact rendered prompt once to verify chat-template formatting.
    tok0 = AutoTokenizer.from_pretrained(BASE_INSTRUCT, trust_remote_code=True, use_fast=True)
    print("=" * 100)
    print("RENDERED PROMPT (enable_thinking=False) for sample 0 — repr of last 400 chars:")
    pt0 = fmt_prompt(tok0, samples[0]["prompt"])
    print(repr(pt0[-400:]))
    print(f"(full prompt length: {len(pt0)} chars)")
    print("=" * 100)

    for label, base, adapter in configs:
        print("\n" + "#" * 100)
        print(f"### MODEL: {label}")
        print("#" * 100)
        torch.manual_seed(0)
        model, tok = load(base, adapter)

        for i, s in enumerate(samples):
            pt = fmt_prompt(tok, s["prompt"])
            ref = s["reference"]
            meta = s["meta"]
            lr = meta.get("length_requirement", "khoảng 50 từ")
            sr = meta.get("sentence_requirement", None)

            print(f"\n----- sample {i} | len_req={lr!r} sent_req={sr!r} -----")
            print(f"REF: {ref}")

            regimes = [
                ("A eval-clean (greedy,no-pen)", dict(train_mode=False, use_cache=True, do_sample=False, penalties=False)),
                ("B rollout-replica (train,sample,pen)", dict(train_mode=True, use_cache=False, do_sample=True, penalties=True)),
                ("C rollout-eval (eval,sample,pen)", dict(train_mode=False, use_cache=True, do_sample=True, penalties=True)),
            ]
            for rlabel, rkw in regimes:
                torch.manual_seed(0)
                g = gen(model, tok, pt, **rkw)
                rd = compute_all_rewards(g, ref, lr, sr)
                print(f"  [{rlabel}] R_acc={rd['accuracy']:.3f} R_len={rd['length']:.3f} R_tot={rd['total']:.3f}")
                print(f"      → {g[:220]!r}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
