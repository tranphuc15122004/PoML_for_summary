# Kế hoạch báo cáo sử dụng kết quả hiện có

> **Trạng thái triển khai:** đã đồng bộ vào `docs/PROJECT_AUDIT.md`, `docs/REPORT_OUTLINE.md`, `docs/report.md`, `docs/problem_statement.md`, `docs/DATASETS.md`, `docs/pipeline_plan.md` và `docs/progress_report.md`.

## Quyết định chung

- Không chạy lại huấn luyện hoặc full evaluation.
- Giữ nguyên các generation đã lưu và coi chúng là kết quả chính thức của project.
- Không ghi decoding là greedy vì code xác nhận run dùng `temperature=0.3`, `top_p=0.9`, `do_sample=True`. Báo cáo trung thực là “one-shot low-temperature decoding, seed không được lưu” và đưa vào limitations.
- V5 là cấu hình chính; v4 chỉ là configuration-bundle ablation.
- ViMs/VLSP và baseline ngoài chỉ mang tính exploratory/contextual.

## Ma trận kết quả chính

Với cả Qwen3 Base và Instruct, trình bày:

1. Pretrained.
2. SFT-only.
3. GRPO-fresh v5.
4. SFT+GRPO v5.

So sánh theo từng dataset và metric:

- ROUGE-2.
- Relative length error.
- Length distance.
- Sentence exact/tolerant hit/MAE.

Kết luận generalization chỉ dựa trên VietNews và WikiLingua. ViMs/VLSP vẫn xuất hiện trong bảng nhưng gắn ký hiệu leakage.

## Phân tích reward hacking v3–v5

Dùng:

- V3: `models/eval_results/20260627_043351`.
- V5: `models/eval_results/20260622_103706`.

Tính lại trên generation đã lưu bằng cùng implementation:

- Degenerate rate.
- Zero-content rate: `R_acc = 0`.
- Near-zero content rate: `R_acc ≤ 0.01`.
- Constraint-only hacking rate: content gần 0 nhưng length/sentence reward cao.
- Mean content reward.
- ROUGE-2 và constraint adherence hiện có.

So sánh bốn cặp tương ứng:

- Base fresh v3 ↔ v5.
- Base SFT-init v3 ↔ v5.
- Instruct fresh v3 ↔ v5.
- Instruct SFT-init v3 ↔ v5.

Chọn 4–6 ví dụ cùng source, trong đó v3 sinh blob/lặp/mất nội dung còn v5 sinh output hợp lệ. Gọi đây là ablation của “reward-hacking mitigation package”, không quy toàn bộ cải thiện cho riêng detector vì còn có các thay đổi hệ thống khác.

## V4–V5 ablation

- V4: `K=4`, `LR=5e-7`, `beta=0.15`.
- V5: `K=8`, `LR=2e-6`, `beta=0.04`.

Báo cáo như độ nhạy với configuration bundle. Không tuyên bố tác động riêng của từng hyperparameter.

## Đánh giá định tính 40 mẫu

Chọn cố định bằng seed 42:

- 15 VietNews.
- 15 WikiLingua.
- 5 ViMs.
- 5 VLSP.

Đánh giá ba output cho mỗi mẫu thuộc họ Instruct:

1. Pretrained.
2. SFT-only.
3. SFT+GRPO v5.

Tổng cộng 120 output, ẩn tên model và xáo trộn thứ tự. Chấm thang 1–5:

- Factuality: mức độ thông tin được source hỗ trợ.
- Fluency: tự nhiên, mạch lạc, không lặp.
- Instruction adherence: tuân thủ độ dài, số câu và định dạng.

Dùng một người chấm và gọi đây là pilot human evaluation; báo cáo mean, median, phân bố điểm và pairwise win rate. Không đưa ra kết luận về inter-annotator agreement.

## Baseline ngoài

- Bảng chính: VietAI, Qwen3-14B, Qwen3-4B, Llama3.3-70B, GPT-4o và hai model SFT+GRPO v5 của project.
- Bảng đầy đủ đưa vào appendix.
- Chỉ so sánh ROUGE-2 và Length Distance trên bốn dataset.
- Không dùng BARTScore của project.
- Ghi rõ baseline ngoài thiếu thông tin đồng nhất về split, prompt, decoding và metric; không tuyên bố vượt trội trực tiếp.

## Chỉnh sửa tài liệu

- Tạo `docs/PROJECT_AUDIT.md`: provenance, ma trận run, validity và phân loại claim.
- Tạo `docs/REPORT_OUTLINE.md`: khung ACL 8 trang và vị trí bảng/hình.
- Cập nhật `problem_statement.md` theo các câu hỏi nghiên cứu mới.
- Cập nhật `report.md` với v5-main, v3–v5 reward analysis và v4–v5 ablation.
- Đồng bộ `DATASETS.md`, `pipeline_plan.md` và `progress_report.md`.
- Giữ `qwen3_training_report.md` như snapshot lịch sử nhưng sửa các diễn giải không còn phù hợp.

## Tiêu chí hoàn thành

- Không mô tả sai evaluation hiện tại là greedy.
- Mọi số liệu chỉ lấy từ artefact đã lưu và có run ID.
- Sentence-loss ablation không nằm trong main paper; chỉ giữ ở appendix với kết luận inconclusive.
- Multi-document và external baseline luôn có cảnh báo validity.
- Contribution chính được giới hạn ở pipeline SFT–GRPO, nghiên cứu Base/Instruct và reward-hacking mitigation package.
