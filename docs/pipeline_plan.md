# Kế hoạch Pipeline Huấn luyện: Tóm tắt Tiếng Việt có Kiểm soát (Length + Persona Control)

> Tài liệu thiết kế cho hướng nghiên cứu 3.1 (Length Control + Persona Control).
> Cập nhật: 2026-06-07. Compute: 1× A100 80GB, PBS walltime 48h/job (NCI Gadi).

---

## 1. Bối cảnh & Vấn đề phát hiện trong code hiện tại

Dự án post-training Qwen2.5-3B-Instruct cho tóm tắt tiếng Việt tuân thủ ràng buộc độ dài và phong cách/persona. Khảo sát codebase hiện tại phát hiện **3 vấn đề nghiêm trọng** phải xử lý trước:

| # | Vấn đề | Vị trí | Hậu quả |
|---|--------|--------|---------|
| 1 | **Custom GRPO trainer hỏng gradient**: logprobs thu bằng `.item()` (tách khỏi computation graph), loss dựng lại từ Python floats | `src/SFT_GRPO/train_grpo.py:523` | `backward()` crash hoặc model không học gì |
| 2 | **Dữ liệu SFT sai thiết kế**: style gán ngẫu nhiên vào instruction nhưng target giữ nguyên văn phong gốc (báo chí) | `src/dataset/augmenter.py:155-190` | Dạy model **bỏ qua** style instruction |
| 3 | **Đếm "từ" thực chất là đếm âm tiết**: `text.split()` đếm tiếng, không phải từ ("nghiên cứu viên" = 3 tokens, 1 từ) | `rewards.py`, `evaluate.py` | Length reward/eval lệch ~40-50% so với số từ thật |

**Quyết định cho vấn đề 3:** GIỮ syllable-level (đếm tiếng) vì toàn bộ văn liệu tiếng Việt (ViT5, BARTpho, VLSP) đều dùng syllable-level ROUGE — nhưng phải ghi rõ trong báo cáo và chuẩn hoá prompt ("từ" hiểu là tiếng/âm tiết).

---

## 2. Kết quả Survey (~40 papers, 4 trục nghiên cứu)

### 2.1. Length Control trong post-training

| Paper | arXiv | Đóng góp chính | Áp dụng |
|-------|-------|----------------|---------|
| LIFT — Following Length Constraints in Instructions (Meta, EMNLP 2025) | 2406.17744 | Augment instruction với length constraint + DPO; violation rate <10% | Khuôn mẫu cho DPO stage |
| Ruler — Meta Length Tokens (2024) | 2409.18943 | Thêm token đặc biệt `<len_50>` vào tokenizer khi SFT; +28 điểm Precise Match | **Dùng trực tiếp** — MLT cho SFT Round B |
| InstructCMP — Length priming (ACL 2024 Findings) | 2406.11097 | Inject gợi ý độ dài vào instruction ("hiện tại X từ → mục tiêu Y từ") | **Dùng trực tiếp** — string ops, gần như free |
| Tulu 3 — RLVR (AI2, 2024) | 2411.15124 | Reward nhị phân từ verifier (1 nếu đúng range, 0 nếu sai), không dùng reward model | **Dùng trực tiếp** — length-hit reward |
| GR³ — Group Relative Reward Rescaling (2026) | 2603.10535 | Reward cộng `quality + λ·length` gây length inflation → 52.9% truncation; phải dùng reward **nhân** (gating) | Sửa công thức reward |
| LARFT — Cognition-Action Gap (2026) | 2603.19255 | Auxiliary task tự đếm độ dài output → +20.9 điểm length-following | Count task trong SFT (LARFT-lite) |
| Precise Length Control — LDPE (2024) | 2412.11937 | Countdown positional encoding, sai số <3 tokens | Tham khảo (cần sửa kiến trúc, không dùng) |
| Zero-Shot Length Control (NAACL 2025) | 2501.00233 | Calibration + filtering không cần train, >90% compliance | Baseline tham chiếu inference-time |

### 2.2. Style/Persona Control

| Paper | arXiv | Đóng góp chính | Áp dụng |
|-------|-------|----------------|---------|
| InstruSum (NAACL 2024 Findings) | 2311.09184 | **Không** LLM-judge nào align tốt với người cho instruction-controllable summarization | Bỏ LLM-judge 3B khỏi RL reward |
| LLM evaluators confuse criteria (ACL 2024) | 2402.12055 | Judge lẫn lộn style với fluency/coherence | Lý do thứ 2 bỏ judge khỏi reward |
| Constraint Back-Translation — CRAB (2024) | 2410.24175 | Gán nhãn constraint mà response **đã thoả mãn sẵn** → data khớp 100% by construction | **Dùng trực tiếp** — sửa lỗi dữ liệu SFT |
| Conifer (2024) | 2404.02823 | Sinh instruction trước → sinh/refine response đến khi thoả constraint → verify | Khuôn mẫu cho rewrite subset |
| Dynamic Multi-Reward Weighting (2024) | 2402.14146 | Style classifier làm discriminator reward trong RL | **Dùng trực tiếp** — PhoBERT classifier reward |
| Readability-controlled summarization (EMNLP 2023) | 2310.10623 | Formula-based reward (Flesch) cho RL | Tham khảo cho proxy đo style |
| MACSum (TACL 2023) | 2211.05041 | Chuẩn dataset multi-attribute: target phải **thực sự** thoả attribute | Nguyên tắc thiết kế data |

### 2.3. RL / Preference Optimization

| Paper | arXiv | Đóng góp chính | Áp dụng |
|-------|-------|----------------|---------|
| Dr. GRPO (2025) | 2503.20783 | GRPO gốc có length bias + difficulty bias; sửa bằng `loss_type="dr_grpo"`, `scale_rewards=False` | **Dùng trực tiếp** — config TRL |
| DAPO (2025) | 2503.14476 | Dynamic sampling (lọc nhóm reward đồng nhất), overlong shaping | Lọc variance≈0 |
| G2D — GRPO-to-DPO (2025) | 2605.21266 | GRPO warmup ngắn (~150-200 steps) → harvest rollout pairs → DPO offline = bằng/hơn pure GRPO với **1/4 compute** | **Xương sống pipeline novel** |
| Scale-Dependent Ranking Inversions (2025) | 2603.19335 | SimPO thất bại thảm ở 3B (dưới cả base model); DPO/GRPO ổn | Tránh SimPO, dùng DPO chuẩn |
| VCRL — Variance-based Curriculum (2025) | 2509.19803 | Reward variance trong nhóm rollout = proxy độ khó → curriculum tự động | Pre-filter prompt cho GRPO |
| SCoRe — Self-Correction RL (ICLR 2025) | 2409.12917 | Train 2 lượt: sinh → sửa, thưởng theo mức cải thiện | Phiên bản "lite": revision task trong SFT |
| VerIF (EMNLP 2025) | 2506.09942 | Hard constraint = code verifier; soft constraint = judge LỚN (QwQ-32B) | Judge 32B chỉ dùng ở eval |
| DeepSeekMath / DeepSeek-R1 | 2402.03300 / 2501.12948 | GRPO gốc; RLVR thuần verifiable tránh reward hacking | Nền tảng lý thuyết |

**TRL GRPOTrainer** (≥ v1.0): hỗ trợ đầy đủ LoRA (`peft_config`) + 1 GPU + custom Python reward funcs + vLLM colocate (`use_vllm=True, vllm_mode="colocate"`). Lưu ý: `sync_ref_model=True` không tương thích LoRA (issue #3108) — giữ default False.

### 2.4. Tiếng Việt — baseline & đặc thù

- **Baseline cần vượt** (syllable-level ROUGE, VietNews): BARTpho R1≈61.1 / R2≈30.3 / RL≈40.2; trần ViT5-large R1≈63.4 / R2≈34.2 / RL≈43.6. WikiLingua: ViT5-large R1≈60.2. VLSP multi-doc: R2-F1 ≈ 0.28-0.30 là cạnh tranh.
- **Chưa có công bố nào áp dụng GRPO/DPO cho tóm tắt tiếng Việt** (GreenMind 2025 dùng GRPO nhưng cho reasoning) → pipeline này có tính novel thực sự.
- BERTScore tiếng Việt: dùng PhoBERT làm backbone (cần word segmentation trước).
- Qwen2.5-3B là lựa chọn hợp lý (multilingual instruction following tốt); Vistral-7B là đối thủ tham chiếu nếu cần.

---

## 3. Thiết kế Pipeline

### 3.1. Pipeline NOVEL: "Prior → Shape → Distill → Repair"

Kết hợp điểm mạnh của 4 phương pháp, mỗi phương pháp tác động một giai đoạn:

```
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 0 — DATA FOUNDATION (nhẹ, tiết kiệm)                         │
│  • Gán nhãn style RẺ (back-translation): VietNews→"báo chí",       │
│    WikiLingua→"sinh hoạt" — không sinh dữ liệu mới                 │
│  • Rewrite CHỈ 15-25K mẫu sang style/persona khác bằng             │
│    Qwen2.5-14B/32B local (vLLM) + verify → loại mẫu fail           │
│  • Persona instructions từ Nemotron-Personas-Vietnam (100K)        │
│  • Inject MLT tokens (<len_50>...) + length priming [Ruler+CMP]    │
│  • Sinh revision triples (draft lỗi → feedback → bản sửa) by rule  │
├────────────────────────────────────────────────────────────────────┤
│ PHASE 1 — SFT 2 ROUND (structural prior)                           │
│  Round A: dữ liệu GỐC chưa augment (~170K) — baseline capability   │
│  Round B: tiếp tục từ A, multi-task:                               │
│    T1: tóm tắt có ràng buộc (MLT + priming, style khớp thật)       │
│    T2: sửa bản tóm tắt theo feedback checker [SCoRe-lite]          │
│    T3: đếm độ dài văn bản [LARFT-lite]                             │
├────────────────────────────────────────────────────────────────────┤
│ PHASE 2 — GRPO WARMUP ~200-300 steps (online RL)                   │
│  • TRL GRPOTrainer: dr_grpo, scale_rewards=False, vLLM colocate    │
│  • R = length_hit(binary) × (w_sem·BERTScore + w_sty·classifier)   │
│    — KHÔNG reward cộng, KHÔNG LLM-judge 3B                         │
│  • Curriculum: tolerance ±20%→±10%, single-doc→multi-doc           │
│  • Lọc nhóm rollout variance≈0 [VCRL]                              │
├────────────────────────────────────────────────────────────────────┤
│ PHASE 3 — G2D: HARVEST → DPO OFFLINE                               │
│  • Sample K=8/prompt từ policy warmup (vLLM)                       │
│  • Checker chấm → cặp high-contrast (chosen≥0.8, rejected≤0.3)     │
│  • DPO trên 5-10K pairs (~1/4 compute pure GRPO)                   │
├────────────────────────────────────────────────────────────────────┤
│ PHASE 4 — VERIFY-REVISE tại inference                              │
│  generate → checker đo → nếu vi phạm: prompt sửa (đã học ở T2)     │
│  → tối đa 1 vòng. Báo cáo 2 điểm: with/without revision            │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2. Công thức reward (Phase 2)

```
R(y) = length_hit(y, req) × [w_sem · semantic(y, ref) + w_sty · style_clf(y, style)]

length_hit  = 1 nếu word_count(y) trong range (±tolerance), 0 nếu ngoài
              + bonus nhỏ 0..0.2 theo độ gần tâm range (gradient mượt)
semantic    = BERTScore-PhoBERT (chính) / ROUGE-L (phụ, log để so sánh)
style_clf   = xác suất đúng style từ PhoBERT classifier
```

Lý do nhân thay vì cộng: chống truncation-hack và length inflation (GR³); output sai độ dài nhận R=0 bất kể chất lượng.

### 3.3. Sử dụng dataset

| Dataset | Vai trò |
|---|---|
| VietNews (~150K, single-doc) | SFT-A (raw) + SFT-B + nguồn rewrite persona |
| WikiLingua (~19.5K, single-doc) | SFT (đa dạng domain how-to) |
| VietNews/WL val (phần dư) | GRPO prompts + harvesting |
| VLSP (~600, multi-doc) | GRPO curriculum giai đoạn khó + test |
| ViMs (300 clusters, multi-doc) | GRPO prompts + test |
| **Nemotron-Personas-Vietnam (100K)** | Map thuộc tính (học vấn→độ phức tạp, nghề→register, tuổi→xưng hô) → persona-conditioned instructions; điều kiện hoá rewriter LLM |

### 3.4. Ma trận thí nghiệm

| Run | Mô tả | Mục đích |
|---|---|---|
| B0 | Qwen2.5-3B zero-shot | sàn |
| B1 | SFT-A (raw data) | đóng góp của SFT cơ bản |
| B2 | SFT-A → GRPO nguyên bản (TRL, reward cộng 0.5·ROUGE+0.3·len+0.2·style, 800 steps) | **baseline pipeline cũ** (đã sửa cho chạy được) |
| N1 | SFT-A → SFT-B → GRPO warmup → DPO → revise | **pipeline novel đầy đủ** |
| A1 | N1 − MLT/priming | ablation prior |
| A2 | N1 − curriculum | ablation curriculum |
| A3 | N1 dừng ở Phase 2 (không DPO) | ablation G2D |
| A4 | N1 − revise | ablation repair |

---

## 4. Các giai đoạn thực hiện

### Stage 1 — Nền tảng + SFT raw (làm ngay)
1. Nâng cấp stack (`requirements.txt`): torch ≥2.5, transformers ≥4.51, **trl ≥1.0**, peft, vllm, datasets ≥3.x → smoke test import.
2. Module mới `src/dataset/persona.py`: load Nemotron-Personas-Vietnam (`load_from_disk`), map thuộc tính → yêu cầu văn phong, sinh instruction.
3. Sửa `src/dataset/augmenter.py`: mode `raw_sft` (không augment), gán nhãn style heuristic, hàm inject MLT+priming.
4. **SFT Round A** trên dữ liệu gốc (~170K, 1 epoch, ước 20-30h) → eval gate: ROUGE gần BARTpho.

### Stage 2 — Data engineering nhẹ + SFT Round B
5. Train PhoBERT style classifier (`src/SFT_GRPO/style_classifier.py`).
6. Rewrite 15-25K mẫu bằng Qwen2.5-14B/32B vLLM + verify (job riêng; đo throughput trước).
7. Sinh revision triples by rule (feedback = template, không cần LLM).
8. **SFT Round B**: mix T1 (constraint+MLT) + T2 (revision) + T3 (count) + persona subset.

### Stage 3 — RL
9. Viết lại `train_grpo.py` trên TRL GRPOTrainer (dr_grpo, vLLM colocate, reward mới trong `rewards.py` — giữ hàm cũ cho baseline B2).
10. **GRPO warmup** 200-300 steps, curriculum + VCRL filter. Multi-doc cần `max_seq_length=4096`.
11. `harvest.py` + `train_dpo.py` (mới): sample K=8 → pairs → **DPO**.
12. Chạy **baseline B2** (reward cộng nguyên bản, 800 steps) để so sánh công bằng.

### Stage 4 — Inference + Đánh giá
13. `revise.py`: vòng verify-revise (tối đa 1 lần sửa).
14. Nâng cấp `evaluate.py`: ROUGE syllable-level + BERTScore-PhoBERT + length MAE/hit-rate theo loại ràng buộc + style classifier score + LLM-judge **Qwen2.5-32B** (chỉ trên ~200 mẫu) + spot-check catastrophic forgetting → bảng so sánh đầy đủ 8 runs.

---

## 5. File thay đổi chính

| File | Thay đổi |
|---|---|
| `requirements.txt` | nâng trl≥1.0, transformers, torch, +vllm, +bert-score |
| `src/dataset/persona.py` | **MỚI** — load & map Nemotron personas |
| `src/dataset/augmenter.py` | mode raw_sft, style labeling, MLT+priming, revision triples |
| `src/SFT_GRPO/rewards.py` | binary length-hit, multiplicative gating, classifier style reward (giữ hàm cũ) |
| `src/SFT_GRPO/train_grpo.py` | viết lại trên TRL GRPOTrainer |
| `src/SFT_GRPO/train_dpo.py` | **MỚI** — DPO stage (G2D) |
| `src/SFT_GRPO/harvest.py` | **MỚI** — rollout sampling + pair construction |
| `src/SFT_GRPO/style_classifier.py` | **MỚI** — train/serve PhoBERT classifier |
| `src/SFT_GRPO/revise.py` | **MỚI** — verify-revise inference |
| `src/SFT_GRPO/evaluate.py` | BERTScore, style clf, judge 32B, forgetting check |
| `scripts/pbs/*` | jobs mới: sft_a, sft_b, grpo_warmup, harvest_dpo, eval |

## 6. Verification

- **Unit tests**: reward functions (mở rộng test trong `rewards.py __main__`), MLT injection, persona mapping.
- **Smoke test mỗi phase** trước job dài: SFT 50 steps/500 mẫu; GRPO 10 steps/32 prompts (loss giảm, backward không crash); DPO 20 steps; đo vLLM throughput trước khi rewrite hàng loạt.
- **Gate giữa stages**: SFT-A đạt ROUGE ≈ BARTpho → mới sang Stage 2; GRPO warmup tăng length hit-rate trên val → mới harvest.
- **End-to-end**: bảng eval 8 runs trên `data/test.jsonl`, lưu `models/eval_results/`.

## 7. Ước lượng compute (calibrate lại sau smoke test)

| Job | Thời gian ước tính |
|---|---|
| SFT-A (170K raw, 1 epoch) | 20-30h |
| Rewrite 15-25K (vLLM 14B/32B) | 4-8h |
| SFT-B (multi-task, ~80-120K) | 8-14h |
| GRPO warmup (200-300 steps) | 6-10h |
| Harvest + DPO | 6-10h |
| Baseline B2 GRPO (800 steps) | 20-30h |
| Eval đầy đủ + ablations | 4h/run |
