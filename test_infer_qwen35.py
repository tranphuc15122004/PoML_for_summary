#!/usr/bin/env python
"""Quick inference test for Qwen3.5-4B."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/g/data/hn98/dd9648/models/Qwen3.5-4B"

print(f"Loading tokenizer from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print(f"Loading model from {MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"Model loaded: {model.config.model_type}, {model.num_parameters()/1e9:.1f}B params")
print(f"Device: {model.device}")
print(f"CUDA available: {torch.cuda.is_available()}")

messages = [
    {"role": "user", "content": "Tóm tắt đoạn văn sau: Việt Nam là một đất nước ở Đông Nam Á, nổi tiếng với văn hóa đa dạng và ẩm thực phong phú."}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

print("\n--- Input ---")
print(input_text[:200])

print("\n--- Generating ---")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n--- Output ---")
print(response)
print("\n✓ Model inference OK!")
