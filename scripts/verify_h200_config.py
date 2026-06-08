"""Quick verify: H200 config produces expected batch/accum values."""
import os, sys
sys.path.insert(0, '/scratch/jp09/dd9648/PoML_for_summary/src')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
from SFT_GRPO.config import SFTConfig, GRPOConfig, ModelConfig

sft = SFTConfig()
grpo = GRPOConfig()

print("=" * 55)
print("H200 Config Verification")
print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)")
print("=" * 55)
print()
print("SFTConfig defaults:")
print(f"  per_device_train_batch_size : {sft.per_device_train_batch_size}")
print(f"  gradient_accumulation_steps : {sft.gradient_accumulation_steps}")
print(f"  effective_batch             : {sft.per_device_train_batch_size * sft.gradient_accumulation_steps}")
print(f"  max_seq_length              : {sft.max_seq_length}")
print(f"  bf16 / fp16                 : {sft.bf16} / {sft.fp16}")
print(f"  gradient_checkpointing      : {sft.gradient_checkpointing}")
print(f"  logging_steps               : {sft.logging_steps}")
print(f"  save_steps                  : {sft.save_steps}")
print()
print("GRPOConfig defaults:")
print(f"  per_device_train_batch_size : {grpo.per_device_train_batch_size}")
print(f"  gradient_accumulation_steps : {grpo.gradient_accumulation_steps}")
print(f"  effective_prompts_per_update: {grpo.per_device_train_batch_size * grpo.gradient_accumulation_steps}")
print(f"  num_generations (K)         : {grpo.num_generations}")
print(f"  max_seq_length              : {grpo.max_seq_length}")
print(f"  max_new_tokens              : {grpo.max_new_tokens}")
print(f"  total_steps                 : {grpo.total_steps}")
print()

# Assertions
assert sft.per_device_train_batch_size == 12, f"SFT batch should be 12, got {sft.per_device_train_batch_size}"
assert sft.gradient_accumulation_steps == 2, f"SFT accum should be 2, got {sft.gradient_accumulation_steps}"
assert sft.max_seq_length == 3072
assert grpo.per_device_train_batch_size == 8, f"GRPO batch should be 8, got {grpo.per_device_train_batch_size}"
assert grpo.gradient_accumulation_steps == 2, f"GRPO accum should be 2, got {grpo.gradient_accumulation_steps}"
assert grpo.max_seq_length == 3072

print("✓ All assertions passed — config correct for H200")
print("=" * 55)
