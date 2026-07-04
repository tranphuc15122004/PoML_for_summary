# Pipeline đã triển khai và cách dùng cho báo cáo hiện tại

**Cập nhật:** 03/07/2026  
**Hạ tầng chính:** NCI Gadi, 1×NVIDIA H200, PBS

## 1. Pipeline hiện tại

```text
Raw datasets
  ├─ VietNews / WikiLingua
  └─ ViMs / VLSP
        │
        ▼
Dataset loaders + normalization
        │
        ▼
Instruction construction
  ├─ SFT: messages + reference
  └─ GRPO: prompt + reference + constraint metadata
        │
        ├───────────────┐
        ▼               ▼
LoRA SFT            GRPO fresh
        │               │
        ▼               │
GRPO SFT-init ◄─────────┘
        │
        ▼
Evaluation + per-sample generations + CSV/JSONL
```

Pipeline chính hiện chỉ học constraint độ dài và số câu. Style/persona không còn là objective thực nghiệm của phiên bản report này.

## 2. Quyết định khóa khi viết báo cáo

- Không chạy lại huấn luyện hoặc full evaluation.
- Giữ nguyên generation đã lưu làm artefact chính thức.
- Không gọi decoding là greedy; mô tả đúng là low-temperature sampling một lần, không lưu seed.
- `v5` là cấu hình chính.
- `v4` là configuration-bundle ablation.
- `v3` là failure-analysis baseline cho reward-hacking mitigation package.

## 3. Canonical runs cần dùng

| Run / file | Vai trò |
|---|---|
| `models/eval_results/20260622_103706` | Main Qwen3 evaluation |
| `models/eval_results/20260627_043351` | Canonical v3 comparison |
| `models/eval_results/20260630_020126` | No-sentence appendix ablation |
| `docs/PROJECT_AUDIT.md` | Claim boundary và provenance |
| `docs/report.md` | Kết quả và diễn giải chính |

## 4. Các quyết định thiết kế đã khóa

### Dữ liệu

- SFT ưu tiên reference chất lượng: loại VietNews title dưới 10 whitespace token.
- GRPO dùng nhiều nguồn hơn, gồm cả VLSP và VietNews title ngắn.
- Constraint lấy từ reference; GRPO luân phiên ba loại template.
- Source/summary truncate theo ký tự ở 8.000/1.500.

### SFT

- LoRA rank/alpha `32/32`, dropout `0.05`.
- LR `5e-5`, 2 epoch, cosine schedule, warmup `0.1`.
- Context `3072`, packing, bf16 trên H200.
- Auto-calibration quyết định batch thực tế.

### GRPO

- Custom loop, không dùng `TRL.GRPOTrainer`.
- K completion được group-normalize reward để tạo advantage.
- Reference là policy tại điểm khởi tạo GRPO.
- Reward có content gate để tránh output đúng constraint nhưng vô nghĩa.
- Không dùng repetition penalty/no-repeat n-gram trong run chính.
- `length_advantage_alpha=0` trong các run báo cáo.

### Evaluation

- Lưu per-sample generation và aggregate CSV/JSON.
- Metrics chính: ROUGE-2, relative length error, length distance.
- Sentence metrics được audit lại từ generation đã lưu.

## 5. Ma trận so sánh của paper

### Main matrix

Với mỗi family `Base` và `Instruct`, chỉ trình bày 4 trạng thái:

1. Pretrained.
2. SFT-only.
3. GRPO-fresh v5.
4. SFT+GRPO v5.

### Supporting analyses

- `v3 -> v5`: reward-hacking mitigation package analysis.
- `v4 -> v5`: configuration-bundle ablation.
- no-sentence ablation: appendix only.

## 6. Việc còn lại trước khi chuyển sang LaTeX

- Chọn 4–6 paired examples `v3 -> v5` cho main text.
- Nếu kịp, triển khai pilot human evaluation 40 mẫu.
- Đóng gói bảng/hình theo khung 8 trang ACL trong `docs/REPORT_OUTLINE.md`.

## 7. Việc có ích nhưng không bắt buộc cho báo cáo hiện tại

- Split sạch cho ViMs/VLSP rồi rerun evaluation.
- Greedy hoặc seeded re-evaluation.
- Chuẩn hóa lại ROUGE dependency để bỏ fallback path.
- Hoàn tất human evaluation nhiều người chấm.

Những việc này sẽ làm paper mạnh hơn nếu có thêm thời gian, nhưng không làm thay đổi decision hiện tại là dùng trung thực các artefact đã lưu.
