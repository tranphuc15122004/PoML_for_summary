# Báo cáo Tiến độ: Post-training Pipeline cho Tóm tắt Văn bản Tiếng Việt

<div align="center">

**Đơn vị:** Viettel AI R&D  
**Dự án:** Hậu huấn luyện (Post-training) mô hình LLM cho tóm tắt văn bản tiếng Việt có ràng buộc  
**Cơ sở hạ tầng:** NCI Gadi HPC — GPU NVIDIA H200 (150 GB VRAM)  
**Ngày cập nhật:** 21/06/2026

</div>

---

## 1. Mục tiêu nghiên cứu

Fine-tune các small LLM để tuân thủ ràng buộc đa chiều khi tóm tắt văn bản tiếng Việt:

| Ràng buộc | Mô tả | Ví dụ prompt |
|:---|---:|---|
| **Độ dài** | Số từ chính xác, khoảng, hoặc không vượt quá | `khoảng 50 từ`, `trong khoảng 40-60 từ`, `không quá 65 từ` |
| **Số câu** | Số câu chính xác hoặc khoảng | `khoảng 2 câu`, `trong khoảng 1-3 câu` |

Pipeline huấn luyện hai giai đoạn: **SFT → GRPO**

---

## 2. Kiến trúc tổng thể

```
VDT_Textsum/              ← Raw datasets (không commit)
├── vietnews-master/
├── WikiLingua/
├── ViMs-Dataset-master/
└── vlsp/

src/
├── dataset/
│   ├── dataset.py          ← Dataset classes (VietNews, WikiLingua, ViMs, VLSP)
│   ├── prepare_no_aug.py   ← Sinh data SFT không augmentation
│   └── augmenter.py        ← Legacy (không còn dùng trong pipeline chính)
├── SFT_GRPO/
│   ├── config.py           ← ModelConfig, SFTConfig, GRPOConfig, EvalConfig
│   ├── train_sft.py        ← SFT trainer (TRL SFTTrainer + LoRA)
│   ├── train_grpo.py       ← Custom GRPO trainer (rollout → reward → PG)
│   ├── rewards.py          ← R_acc (ROUGE-L) + R_len + R_sent
│   └── metrics_logger.py   ← CSV/JSONL logging
└── evaluation/
    └── evaluate.py         ← Evaluation pipeline (ROUGE-2, Length Error, BARTScore)
```

---

## 3. Dữ liệu

### 3.1 Nguồn dữ liệu thô

| Dataset | Loại | Số mẫu | Ghi chú |
|:---|---:|---:|---|
| **VietNews** | Single-doc abstractive | ~130K train / ~10K val / ~10K test | Tin tức tiếng Việt; title làm target |
| **WikiLingua** | Single-doc abstractive | ~16K train / val / test | Vietnamese WikiHow |
| **ViMs** | Multi-doc abstractive | 300 clusters | 5–10 docs/cluster, 2 gold summaries |
| **VLSP** | Multi-doc extractive | 285 train / 15 val / 200 test | VLSP 2022 AbMuSu |

### 3.2 Pipeline sinh data hiện tại (không augmentation)

Sau khi loại bỏ cơ chế augmentation, data được sinh qua `prepare_no_aug.py` với format đơn giản hóa: mỗi bài sinh **một mẫu duy nhất**, constraints (độ dài, số câu) được lấy trực tiếp từ gold summary.

#### Format SFT (`data/sft_train.jsonl`)
```json
{
  "messages": [
    {"role": "system", "content": "Bạn là một trợ lý AI chuyên tóm tắt văn bản tiếng Việt. Hãy tạo ra bản tóm tắt ngắn gọn, chính xác, tuân thủ đúng yêu cầu về độ dài và số câu."},
    {"role": "user",   "content": "Yêu cầu:\n- Độ dài: khoảng 22 từ\n- Số câu: khoảng 1 câu\n\nVăn bản:\n{source}"},
    {"role": "assistant", "content": "{target}"}
  ],
  "meta": {"source_length": ..., "target_length": ...}
}
```

**Format GRPO** (`data/grpo_train.jsonl`):
```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "Yêu cầu:\n- Độ dài: trong khoảng 12-18 từ\n- Số câu: trong khoảng 1-2 câu\n\nVăn bản:\n{source}"}
  ],
  "reference": "{gold_summary}",
  "meta": {"length_requirement": "...", "sentence_requirement": "..."}
}
```

### 3.3 Kích thước data hiện tại

| File | Số dòng | Ghi chú |
|:---|---:|---|
| `data/sft_train.jsonl` | **111,150** | VietNews train + WikiLingua train |
| `data/sft_val.jsonl` | 2,500 | 2K VietNews val + 500 WikiLingua val |
| `data/grpo_train.jsonl` | **119,657** | Prompts GRPO không có assistant |
| `data/grpo_val.jsonl` | 21,882 | |
| `data/test.jsonl` | 2,800 | 2K VN + 500 WL + 300 ViMs |

> **So sánh với pipeline cũ:** Trước khi loại bỏ augmentor, `sft_train.jsonl` có 358,251 mẫu (×3 variants/bài, với style/persona phức tạp). Hiện tại 111,150 mẫu (×1/bài, chỉ length + sentence constraints).

---

## 4. Cấu hình mô hình & huấn luyện

### 4.1 LoRA (dùng chung cho SFT và GRPO)

| Tham số | Giá trị |
|:---|---:|
| Rank (`r`) | 32 |
| Alpha | 32 (scaling = 1) |
| Dropout | 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |

### 4.2 SFT Config (mặc định)

| Tham số | H200 | A100-80G | V100/≤32GB |
|:---|---:|---:|---:|
| Batch/device | 16 | 8 | 1 |
| Grad accum | 1 | 2 | 16 |
| Effective batch | **16** | **16** | **16** |
| Max seq len | 3072 | 3072 | 2048 |
| Learning rate | 5e-5 | 5e-5 | 5e-5 |
| Epochs | 2.0 | 2.0 | 2.0 |
| LR scheduler | cosine | cosine | cosine |
| Warmup ratio | 0.1 | 0.1 | 0.1 |
| Dtype | bf16 | bf16 | fp16 |
| Packing | ✓ | ✓ | ✗ |
| Grad checkpointing | ✗ | ✗ | ✓ |

> **Auto-calibration:** Trước khi train, `calibrate_batch_size()` chạy probe 2 lần forward+backward để tìm batch tối ưu không OOM (target 90% VRAM).

### 4.3 GRPO Config

Config thực tế được phân chia theo version:

| Tham số | v4 (hiện tại) | v5 (tối ưu) | Ghi chú |
|:---|---:|---:|---|
| Num generations (K) | 4 | **8** | v5: giảm variance advantage |
| Temperature | 0.7 | 0.7 | |
| Max new tokens | 80 | 80 | reference p95=32 tokens, 80 đủ buffer |
| Batch/device (H200) | 6 (auto-cal) | auto-cal | |
| Grad accum (H200) | 3 (auto-cal) | auto-cal | |
| Learning rate | 5e-7 | **2e-6** | v5: 4× lớn hơn |
| LR scheduler | constant | constant | |
| Warmup steps | 20 | 20 | |
| Epsilon (PPO clip) | 0.2 | 0.2 | |
| Beta (KL penalty) | 0.15 | **0.04** | v4 dùng 0.15 vì 0.04 gây KL ≈ 4.6 trong v3 |
| Total steps | 800 | 800 | |
| Dtype | bf16 | bf16 | |
| Repetition penalty | **1.0** | 1.0 | PHẢI giữ 1.0 — 1.3 phá huỷ R_acc |
| No-repeat ngram size | **0** | 0 | PHẢI giữ 0 — ngram=3 cấm dùng trigram từ source |
| Disable thinking | True | True | Qwen3 family |

---

## 5. Hàm Reward (GRPO)

### 5.1 Ba thành phần

**R_acc — Accuracy (ROUGE-L F1)**

```
R_acc = 2 × LCS(gen, ref) / (|gen| + |ref|)
```

- Range `[0, 1]`
- LCS tính ở mức syllable (split by space)

**R_len — Length Adherence**

| Loại constraint | Logic tính điểm |
|:---|---|
| `khoảng X từ` | Tolerance ±20%; linear decay từ tol → 3×tol |
| `trong khoảng lo-hi từ` | Full score trong `[lo, hi]`; proportional decay ngoài range |
| `không quá X từ` | Full score nếu ≤ X; linear decay nếu vượt |

**R_sent — Sentence Count Adherence**

- Đếm câu bằng regex `[.!?]` (bỏ qua số thập phân)
- Tolerance ±1 cho `khoảng X câu`

### 5.2 Anti-Reward-Hacking

Phát hiện trong quá trình chạy baseline: model học cách **game** reward bằng cách sinh văn bản thoái hóa (blob không khoảng trắng, lặp token) để nhận `R_len=1.0` và `R_sent=1.0` với `R_acc=0`, tổng reward `=0.5` thay vì phải tóm tắt thực sự.

**Cơ chế phát hiện `_is_degenerate(text)`:**

| Điều kiện | Ngưỡng | Phát hiện |
|:---|---|:---|
| Chuỗi không khoảng trắng > 25 ký tự | blob detector | 🔴 |
| Số ký tự alpha < 3 | output rỗng/ký hiệu | 🔴 |
| Alpha ratio < 12% (nếu > 15 ký tự) | không phải text | 🔴 |
| Tần suất từ phổ biến nhất > 60% (≥5 từ) | repetition loop | 🔴 |
| Unique token ratio < 30% (≥5 từ) | low diversity | 🔴 |

> **Floor cho `không quá X từ`:** Yêu cầu tối thiểu `max(3, 15% × X)` từ. Output 1-2 từ không còn nhận `R_len=1.0`.
>
> Nếu `_is_degenerate() = True`: `R_len = R_sent = 0.0`, chỉ còn `0.5 × R_acc`.

### 5.3 Composite Reward

```
R_total = 0.5 × R_acc + 0.3 × R_len + 0.2 × R_sent    (nếu có sentence req)
R_total = (0.5 × R_acc + 0.3 × R_len) / 0.8            (nếu không có)
```

### 5.4 GRPO Training Loop

```
For each step:
  1. Rollout: sinh K completions / prompt bằng policy model (eval mode — LoRA dropout off)
  2. Reward: tính R_acc, R_len, R_sent cho mỗi completion
  3. Advantage: group normalization
       A = (R - mean(R_group)) / max(std(R_group), 0.01)
       Length scaling: disabled (alpha=0.0)  ← bật lại sau khi R_acc ổn định
  4. Policy gradient loss (on-policy):
       old_logprobs = new_logprobs.detach()  ← π_old = π_θ, ρ = 1 tại thời điểm update
       L_pg = -E[min(ρ·A, clip(ρ, 1-ε, 1+ε)·A)]
  5. KL penalty (per-token): L_kl = β × KL(π || π_ref)
       KL_token = exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1
  6. Total: L = L_pg + L_kl
```

> **Lưu ý triển khai quan trọng:**
> - Generation dùng `eval()` mode để tắt LoRA dropout — tránh off-policy mismatch giữa rollout và forward pass tính log-prob
> - `old_logprobs = new_logprobs.detach()` → on-policy training, clipping không kích hoạt thực tế (`ρ=1` luôn) nhưng gradient = `-advantage` per token là đúng
> - Reference model luôn frozen, không qua `accelerator.prepare()`

---

## 6. Các lần chạy đã hoàn thành

### 6.1 Qwen2.5-3B-Instruct — SFT

| Run | Dataset | Ngày | Thời lượng | Steps | Status |
|:---|---:|---:|---:|---:|:---|
| `sft_aug_Qwen25_3B_2e` | SFT augmented (358K) | 09/06/2026 | **19h 46m** | ~9,356 | ✅ Hoàn thành |
| `sft_no_aug_Qwen25_3B_2e` | No-aug (119K) | 09/06/2026 | **5h 27m** | ~2,998 | ✅ Hoàn thành |

> Peak GPU memory: 55.36 GB / 150 GB H200.

### 6.2 Qwen3.5-4B — SFT (pipeline cũ, có augmentation)

| Run | Dataset | Ngày bắt đầu | Checkpoint cuối | Status |
|:---|---:|---:|---:|:---|
| `sft_aug_Qwen3.5-4B` | SFT augmented (358K) | 10/06/2026 | `checkpoint-3000` (~epoch 0.55) | ⚠️ Bị gián đoạn |
| `sft_no_aug_Qwen3.5-4B` | No-aug (119K) | 09/06/2026 | `checkpoint-3200` (~epoch 1.90) | ⚠️ Bị gián đoạn |

> Cả hai run bị restart nhiều lần do lỗi môi trường, chưa tạo được `final` checkpoint.

### 6.3 Qwen3-4B (Base & Instruct) — SFT

Đã hoàn thành 17–18/06/2026:

| Job | Model | Checkpoint cuối | Eval Loss | Status |
|:---|---:|---:|:---:|:---|
| SFT Base | Qwen3-4B-Base | `sft_qwen3_4b_base/final` | Hội tụ epoch 2 | ✅ Hoàn thành |
| SFT Instruct | Qwen3-4B | `sft_qwen3_4b_instruct/final` | Hội tụ epoch 2 | ✅ Hoàn thành |

> Cả hai hội tụ hoàn toàn tại epoch 2 (Δeval_loss < 0.0005 trong 400 step cuối). Không cần train thêm epoch.

### 6.4 Qwen3-4B — GRPO Baseline (pre-fix, làm so sánh)

Submit 19/06/2026, chạy với config **chưa có fix** (không repetition_penalty, reward cũ):

| Job ID | Variant | Output dir | Steps | R_acc cuối |
|:---|---:|:---|---:|---:|
| 171632423 | base + fresh | `grpo_qwen3_4b_base_fresh` | 400 | ~0.0004 |
| 171632426 | base + sft init | `grpo_qwen3_4b_base_sft` | 400 | ~0.0053 |
| 171632427 | instruct + fresh | `grpo_qwen3_4b_instruct_fresh` | 800 | ~0.0000 |
| 171632428 | instruct + sft init | `grpo_qwen3_4b_instruct_sft` | 800 | ~0.0048 |

> **Chẩn đoán:** R_acc ≈ 0 toàn bộ do rollout thoái hóa (model sinh blob/loop lặp → ROUGE-L=0). Reward hacking: blob "Is 111..." nhận R_len=1.0 + R_sent=1.0, tổng reward=0.5 mà không có nội dung.

### 6.5 Qwen3-4B — GRPO v2 (hoàn thành)

Submit 20/06/2026 với đầy đủ các fix chống degeneration và reward hacking:

| Job ID | Variant | Output dir | Steps | Status |
|:---|---:|:---|---:|:---|
| 171721345 | instruct + SFT init | `grpo_qwen3_4b_instruct_sft_v2` | 800 | ✅ Hoàn thành |
| 171721346 | base + SFT init | `grpo_qwen3_4b_base_sft_v2` | 800 | ✅ Hoàn thành |
| 171721347 | instruct + fresh | `grpo_qwen3_4b_instruct_fresh_v2` | 800 | ✅ Hoàn thành |
| 171721348 | base + fresh | `grpo_qwen3_4b_base_fresh_v2` | 800 | ✅ Hoàn thành |

> **Smoke test xác nhận (20/06/2026, job 171719939):** 22/22 PASS.  
> Step 1: `R_acc = 0.2365`, `R_mean = 0.4158` — tăng **44×** so với baseline.
>
> **Cấu hình PBS:** `gpuhopper`, 1 GPU H200, 12 CPU, 60 GB RAM, walltime 24h.

---

### 6.6 Qwen3-4B — GRPO v3 (rollout decoding bug → R_acc≈0)

Submit sau v2, thử nghiệm thêm `repetition_penalty=1.3` và `no_repeat_ngram_size=3` để chống degenerate output:

| Variant | Output dir | R_acc (step 400) | Vấn đề |
|:---|---:|---:|:---|
| base + fresh | `grpo_qwen3_4b_base_fresh_v3` | ≈ 0.000 | 🔴 Rollout degenerate |
| base + sft | `grpo_qwen3_4b_base_sft_v3` | ≈ 0.000 | 🔴 Rollout degenerate |
| instruct + fresh | `grpo_qwen3_4b_instruct_fresh_v3` | ≈ 0.000 | 🔴 Rollout degenerate |
| instruct + sft | `grpo_qwen3_4b_instruct_sft_v3` | ≈ 0.000 | 🔴 Rollout degenerate |

> **Chẩn đoán:** `repetition_penalty > 1.0` và `no_repeat_ngram_size > 0` áp dụng lên toàn bộ `input_ids` (kể cả prompt ~2000 tokens). Model bị phạt khi reuse vocab từ bài gốc → summary không thể dùng từ ngữ bài báo → ROUGE-L = 0. Đây là **lỗi thiết kế của HuggingFace `generate()`**, không phải lỗi model.
>
> **Quyết định:** Revert về `repetition_penalty=1.0`, `no_repeat_ngram_size=0`. Degenerate output xử lý hoàn toàn reward-side qua `_is_degenerate()`.

---

### 6.7 Qwen3-4B — GRPO v4 (đã sửa, đang chạy — flat learning)

Submit 20/06/2026 sau khi revert v3, config sạch:

| Job ID | Variant | Output dir | R_acc TB | KL (step 200) |
|:---|---:|:---|---:|---:|
| 171794033 | base + fresh | `grpo_qwen3_4b_base_fresh_v4` | ~0.27 | ~0.004 |
| 171794034 | base + sft | `grpo_qwen3_4b_base_sft_v4` | ~0.27 | ~0.004 |
| 171794035 | instruct + sft | `grpo_qwen3_4b_instruct_sft_v4` | ~0.27 | ~0.004 |
| 171794036 | instruct + fresh | `grpo_qwen3_4b_instruct_fresh_v4` | ~0.27–0.33 | 0.0046 |

**Phân tích học (step 1–200, `instruct_fresh_v4` làm đại diện):**

| Metric | Thay đổi thực (200 steps) | Noise (std) | Signal / Noise |
|---|---|---|---|
| R_acc | +0.0034 | ±0.0336 | **0.10** (quá thấp) |
| R_mean | +0.0074 | ~0.04 | 0.18 |
| KL | 0.0006 → 0.0046 | — | 7.7× tăng |

> **Kết luận:** Model **không học được** có ý nghĩa thống kê. Noise lớn gấp 10× signal. KL tăng (policy đang thay đổi) nhưng reward không theo.
>
> **Nguyên nhân gốc rễ:**
> 1. `num_generations=4` — advantage estimator có variance cao (chỉ 3 bậc tự do)
> 2. `learning_rate=5e-7` — update quá nhỏ, không vượt được gradient noise
> 3. `beta=0.15` — KL penalty kìm policy quá mạnh (KL thực chỉ 0.005)

---

### 6.8 Qwen3-4B — GRPO v5 (config tối ưu)

Submit 21/06/2026 với config điều chỉnh dựa trên phân tích v4:

| Job ID | Variant | Output dir | Config thay đổi |
|:---|---:|:---|---|
| 171836659 | base + fresh | `grpo_qwen3_4b_base_fresh_v5` | K=8, LR=2e-6, β=0.04 |
| 171836660 | base + sft | `grpo_qwen3_4b_base_sft_v5` | K=8, LR=2e-6, β=0.04 |
| 171836662 | instruct + fresh | `grpo_qwen3_4b_instruct_fresh_v5` | K=8, LR=2e-6, β=0.04 |
| 171836663 | instruct + sft | `grpo_qwen3_4b_instruct_sft_v5` | K=8, LR=2e-6, β=0.04 |

**Lý do thay đổi:**

| Tham số | v4 → v5 | Tác động |
|:---|---:|---|
| K (num generations) | 4 → **8** | Giảm variance advantage ~40% (std ∝ 1/√K) |
| Learning rate | 5e-7 → **2e-6** (×4) | 4× lớn hơn, vẫn conservative với GRPO LoRA |
| Beta (KL penalty) | 0.15 → **0.04** | Phù hợp với mức KL thực tế; 0.15 kìm không cần thiết |

> **Walltime:** 24h (với K=8, ước ~55s/step → 800 steps ≈ 12.2h, vẫn trong walltime).  
> **PBS script cập nhật:** Thêm `TEMPERATURE` env var để override qua `qsub -v`.

---

## 7. Kết quả Evaluation (baseline)

Chạy trên 3,100 mẫu test, model sinh với `temperature=0.3`, `max_new_tokens=256`:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Len Hit Rate | AvgLen | BARTScore | G-Eval |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B base (zero-shot) | 0.108 | 0.037 | 0.071 | 21.4% | 75.2 | -3.44 | 0.099 |
| Qwen2.5-3B + SFT aug | 0.130 | 0.055 | 0.093 | 25.2% | 127.3 | -3.48 | 0.049 |
| Qwen2.5-3B + SFT no-aug | 0.108 | 0.042 | 0.082 | 23.5% | 64.9 | -3.82 | 0.050 |
| Qwen3.5-4B base (zero-shot) | 0.086 | 0.040 | 0.059 | 18.2% | 71.5 | **-1.27** | **0.466** |

**Nhận xét:**

- SFT aug cải thiện ROUGE-L (+31%) và length hit rate so với base Qwen2.5-3B
- SFT aug làm model sinh dài hơn nhiều (127 vs 75 từ) → length error tăng
- Qwen3.5-4B base có G-Eval và BARTScore cao vượt trội (chất lượng ngôn ngữ tốt hơn) nhưng ROUGE thấp — do không match gold reference style
- **Length hit rate 18–25%** là điểm yếu chính → GRPO nhắm vào đây

---

## 8. Module Evaluation

File: `src/evaluation/evaluate.py`

**Metrics được tính:**

| Metric | Công thức / Nguồn |
|:---|---|
| **ROUGE-2 F1** | Word-level n-gram overlap |
| **Length Error %** | `|gen_words - required_words| / required_words × 100` |
| **BARTScore** | `log P(generation | article)`, cần `facebook/bart-large-cnn` |

**Cách chạy:**

```bash
# Eval nhanh (không BARTScore)
PYTHONPATH=src python src/evaluation/evaluate.py \
    --models "base=/path/to/model,sft=models/sft_lora/final" \
    --test_data data/test.jsonl \
    --no_bart_score

# Full eval qua PBS
qsub scripts/pbs/eval.pbs
```

---

## 9. Môi trường & Hạn chế Kỹ thuật

### 9.1 Flash Attention

**flash_attn không khả dụng** trên Gadi với PyTorch 2.12.0+cu130:

| Phương án | Kết quả | Nguyên nhân |
|:---|:---:|:---|
| flash_attn 2.7.4 (pre-installed) | ❌ ImportError | C++ ABI mismatch; CUDA symbol mismatch |
| Build 2.8.3.post1 từ source | ❌ Build fail | Cần CUDA 13.0, Gadi chỉ có 12.9 |
| Pre-built wheel 2.8.3 (cu13torch2.9) | ❌ ImportError | Cần GLIBC 2.32, Gadi có 2.28 (RHEL 8) |

> **Fallback:** SDPA (`scaled_dot_product_attention`), chậm hơn ~10-15% nhưng kết quả đúng. Smoke test và training hoạt động bình thường.

### 9.2 Quản lý tài nguyên HPC

| Project | Quota Q2 2026 | Còn lại (ước tính) | Ghi chú |
|:---|---:|---:|:---|
| `hn98` | 1.77 MSU | ~1.18 KSU | 🔴 Hết, mở lại 01/07/2026 |
| `li96` | 450 KSU | ~50 KSU | 🟡 4 GRPO v2 jobs đang chạy (~13 SU/job) |
| `jp09` | 9 KSU | 328 SU | 🔴 Không đủ cho job GPU |

> Model weights: `/g/data/hn98/dd9648/models/` (gdata, không bị quota compute).

---

## 10. Bước tiếp theo

### 🎯 Ưu tiên ngay

| # | Công việc | Chi tiết |
|:---:|:---|---|
| **1** | **Theo dõi v5 jobs** (171836659–663) | Dấu hiệu học tốt sau step 50–100: KL > 0.01, R_acc slope > 0.001/step, S/N ratio > 0.3. Nếu vẫn flat: thử `LR=5e-6` hoặc `K=12` |
| **2** | **Dọn dẹp models** | Xóa: `grpo_qwen3_4b_*_v3` (4 dirs ~15G) + `grpo_validate_fix` (1.5G). Giữ: v4, v5, SFT finals |
| **3** | **Chạy eval so sánh** | `qsub scripts/pbs/eval.pbs` — metrics: ROUGE-L, Length Hit Rate, BARTScore |

### 📋 Khi v5 hoàn thành

| # | Công việc |
|:---:|---|
| **4** | **So sánh v4 vs v5** — xác nhận K=8 + LR=2e-6 + β=0.04 cải thiện learning curve |
| **5** | Kích hoạt `length_advantage_alpha > 0` (hiện tại = 0.0) để amplify completion đúng độ dài |
| **6** | **Xây dựng eval pipeline đầy đủ** — so sánh SFT → GRPO v4 → GRPO v5 vs baseline |

### 💾 Tài nguyên

- `hn98` quota Q3 mở từ **01/07/2026** — có thể dùng cho eval jobs và experiments tiếp theo
- v3 dirs: giải phóng ~15G trên scratch sau khi xác nhận v4/v5 không cần làm baseline so sánh
