#!/usr/bin/env python
"""Legacy decoding-matrix experiment: isolate the exact cause of GRPO rollout garbage.

Loads the SFT checkpoint (the realistic GRPO start policy, LoRA UNMERGED so dropout
layers exist) and generates on N val prompts under a matrix of decoding configs.
Reports per-config mean R_acc for historical decoding settings; this is not the canonical v5 rollout.
and validates the proposed fix (TRL-style: no penalties, eval mode).

Configs (eval mode unless _TRAIN):
  1 greedy_nopen        greedy, no penalty                      → clean reference quality
  2 sample_nopen        T=0.7 top_p=0.9, no penalty             → TRL-style rollout = PROPOSED FIX
  3 greedy_pen          greedy + rep=1.3 + no_repeat=3          → = historical penalty probe
  4 sample_pen          T=0.7 + rep=1.3 + no_repeat=3           → = historical rollout probe (single prompt)
  5 sample_rep11        T=0.7 + rep=1.1 only                    → mild penalty
  6 sample_nopen_TRAIN  T=0.7 no penalty, model.train()         → isolate LoRA-dropout alone
  7 rollout_replica     T=0.7 + rep=1.3 + no_repeat=3, TRAIN, BATCHED(left-pad) → full real rollout
"""

from __future__ import annotations

import json
import os
import sys
import types

if "triton.ops" not in sys.modules:
    _o = types.ModuleType("triton.ops")
    _p = types.ModuleType("triton.ops.matmul_perf_model")
    _p.early_config_prune = lambda *a, **kw: None
    _p.estimate_matmul_time = lambda *a, **kw: 0.0
    sys.modules["triton.ops"] = _o
    sys.modules["triton.ops.matmul_perf_model"] = _p

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = "/scratch/jp09/dd9648/PoML_for_summary"
sys.path.insert(0, os.path.join(PROJECT, "src"))
from SFT_GRPO.rewards import compute_all_rewards  # noqa: E402

BASE = "/g/data/hn98/dd9648/models/Qwen3-4B"
SFT = f"{PROJECT}/models/sft_qwen3_4b_instruct/final"
VAL = f"{PROJECT}/data/grpo_val.jsonl"
N = 5


def fmt(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def gen_single(model, tok, ptext, *, train, do_sample, rep, nrng):
    model.train() if train else model.eval()
    model.config.use_cache = True
    inp = tok(ptext, return_tensors="pt", truncation=True, max_length=3072 - 80).to(model.device)
    kw = dict(max_new_tokens=80, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    kw.update(dict(do_sample=True, temperature=0.7, top_p=0.9) if do_sample else dict(do_sample=False))
    if rep != 1.0:
        kw["repetition_penalty"] = rep
    if nrng:
        kw["no_repeat_ngram_size"] = nrng
    with torch.no_grad():
        out = model.generate(**inp, **kw)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()


def gen_batched(model, tok, ptexts, *, train, rep, nrng):
    """Replicate the real rollout: model.train(), left-padded BATCH, sampling + penalties."""
    model.train() if train else model.eval()
    model.config.use_cache = True
    inp = tok(ptexts, return_tensors="pt", padding=True, truncation=True, max_length=3072 - 80).to(model.device)
    kw = dict(max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9,
              pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    if rep != 1.0:
        kw["repetition_penalty"] = rep
    if nrng:
        kw["no_repeat_ngram_size"] = nrng
    with torch.no_grad():
        out = model.generate(**inp, **kw)
    plen = inp.input_ids.shape[1]
    return [tok.decode(o[plen:], skip_special_tokens=True).strip() for o in out]


def main():
    samples = []
    with open(VAL) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= N:
                break

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, SFT)  # unmerged → dropout present

    ptexts = [fmt(tok, s["prompt"]) for s in samples]

    def racc(gen, s):
        return compute_all_rewards(gen, s["reference"], s["meta"].get("length_requirement", "khoảng 50 từ"),
                                   s["meta"].get("sentence_requirement"))["accuracy"]

    single_configs = [
        ("1 greedy_nopen      ", dict(train=False, do_sample=False, rep=1.0, nrng=0)),
        ("2 sample_nopen (FIX)", dict(train=False, do_sample=True, rep=1.0, nrng=0)),
        ("3 greedy_pen        ", dict(train=False, do_sample=False, rep=1.3, nrng=3)),
        ("4 sample_pen        ", dict(train=False, do_sample=True, rep=1.3, nrng=3)),
        ("5 sample_rep11      ", dict(train=False, do_sample=True, rep=1.1, nrng=0)),
        ("6 sample_nopen_TRAIN", dict(train=True, do_sample=True, rep=1.0, nrng=0)),
    ]

    print("=" * 100)
    print(f"DECODING MATRIX on SFT/final, {N} val samples. Per-config mean R_acc + sample-0 output.")
    print("=" * 100)
    results = {}
    for label, kw in single_configs:
        accs, ex0 = [], None
        for i, (pt, s) in enumerate(zip(ptexts, samples)):
            torch.manual_seed(0)
            g = gen_single(model, tok, pt, **kw)
            accs.append(racc(g, s))
            if i == 0:
                ex0 = g
        mean = sum(accs) / len(accs)
        results[label.strip()] = mean
        print(f"\n[{label}] mean R_acc = {mean:.3f}   per-sample={['%.2f'%a for a in accs]}")
        print(f"     sample0 → {ex0[:200]!r}")

    # Config 7: full rollout replica — batched, train mode, penalties
    print("\n" + "-" * 100)
    torch.manual_seed(0)
    gens = gen_batched(model, tok, ptexts, train=True, rep=1.3, nrng=3)
    accs = [racc(g, s) for g, s in zip(gens, samples)]
    mean = sum(accs) / len(accs)
    results["7 rollout_replica (batched,train,pen)"] = mean
    print(f"[7 rollout_replica (BATCHED,train,pen)] mean R_acc = {mean:.3f}   per-sample={['%.2f'%a for a in accs]}")
    for i, g in enumerate(gens[:3]):
        print(f"     sample{i} → {g[:200]!r}")

    print("\n" + "=" * 100)
    print("SUMMARY (mean R_acc, higher=better):")
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        print(f"   {v:.3f}   {k}")
    print("=" * 100)


if __name__ == "__main__":
    main()
