# Báo cáo tiến độ project

**Cập nhật:** 03/07/2026  
**Trạng thái chung:** pha thực nghiệm lõi đã khóa; hiện đang đóng gói evidence để viết báo cáo/paper.

## 1. Quyết định chốt ở thời điểm này

- Không chạy lại huấn luyện hoặc full evaluation.
- Dùng generation đã lưu làm kết quả chính thức của project.
- `v5` là cấu hình chính.
- `v4` là configuration-bundle ablation.
- `v3` dùng cho failure analysis của reward-hacking mitigation package.
- ViMs/VLSP và baseline ngoài chỉ mang tính exploratory/contextual.

## 2. Hạng mục đã khóa

| Hạng mục | Trạng thái |
|---|---|
| Pipeline dữ liệu -> SFT -> GRPO -> eval | Hoàn thành |
| Ma trận Qwen3 Base/Instruct | Hoàn thành |
| Main evaluation run `20260622_103706` | Hoàn thành |
| Canonical v3 comparison `20260627_043351` | Hoàn thành |
| No-sentence ablation `20260630_020126` | Hoàn thành |
| Leakage audit | Hoàn thành |
| Reward-hacking audit trên generation đã lưu | Hoàn thành |
| Tài liệu nguồn cho paper | Đang hoàn thiện |

## 3. Kết luận khoa học hiện tại

| Kết luận | Bằng chứng |
|---|---|
| SFT tạo bước nhảy lớn nhất | Base `0.1056 -> 0.2632`, Instruct `0.1506 -> 0.2609` ở ROUGE-2 combined |
| GRPO cần SFT-init | Các nhánh `fresh` vẫn kém rõ so với `SFT` và `SFT+GRPO` |
| Best quality hiện có | `Qwen3 Instruct + SFT + GRPO v5` |
| V5 là cấu hình main hợp lý | thắng rõ ở nhánh `fresh`, mạnh nhất ở nhánh Instruct SFT-init |
| Reward-hacking mitigation package có tác dụng | `Base fresh` giảm degenerate rate `23.81% -> 0.32%` từ `v3 -> v5` |
| Sentence reward riêng lẻ chưa kết luận | No-sentence ablation hiện còn inconclusive |

## 4. Provenance nên dùng khi viết

- `docs/PROJECT_AUDIT.md`: nguồn chuẩn cho provenance, validity và claim boundary.
- `docs/report.md`: bảng kết quả và diễn giải chính.
- `docs/REPORT_OUTLINE.md`: khung ACL 8 trang và vị trí bảng/hình.
- `docs/qwen3_training_report.md`: snapshot lịch sử, không dùng làm nguồn claim cuối cùng nếu mâu thuẫn với audit hiện tại.

## 5. Việc còn lại trước khi chuyển sang LaTeX

- Chọn 4–6 paired examples `v3 -> v5` cuối cùng cho main text.
- Nếu kịp, hoàn thành pilot human evaluation 40 mẫu.
- Chuyển bảng/hình từ Markdown sang `FormalReport_VDT/latex/` theo outline đã khóa.

## 6. Limitations bắt buộc nêu trong báo cáo

1. Evaluation hiện tại là one-shot low-temperature sampling, không phải greedy, và không lưu seed.
2. ViMs/VLSP có train-test leakage nên chỉ được dùng exploratory.
3. Baseline ngoài project không cùng protocol nên không được dùng để tuyên bố vượt trội trực tiếp.
4. BARTScore của run chính là `NaN`.
5. SFT dùng TRL `0.13`, nên assistant-only masking không hoạt động như giả định ban đầu.

## 7. Kết luận về tiến độ

Project đã đủ evidence cho một báo cáo project mạnh nếu giữ kỷ luật diễn giải. Phần còn lại chủ yếu là đóng gói, chọn ví dụ và chuyển narrative sang LaTeX, không phải mở rộng thêm một vòng thực nghiệm lớn.
