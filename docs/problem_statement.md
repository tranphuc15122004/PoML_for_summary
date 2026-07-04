# Phát biểu bài toán: hậu huấn luyện cho tóm tắt tiếng Việt có kiểm soát

**Đơn vị:** Viettel AI  
**Backbone chính của báo cáo:** Qwen3-4B-Base và Qwen3-4B-Instruct  
**Phiên bản report hiện tại:** dùng artefact đã lưu, không rerun toàn bộ huấn luyện/evaluation

## 1. Bối cảnh

Các ứng dụng tóm tắt trong trợ lý ảo, giám sát thông tin và hệ thống doanh nghiệp không chỉ cần bản tóm tắt đúng nội dung mà còn phải tuân thủ giới hạn đầu ra. Với tiếng Việt, mô hình nhỏ và mô hình open-weight thường:

- sinh quá dài hoặc quá ngắn so với yêu cầu;
- tuân thủ số câu không ổn định;
- dễ đánh đổi chất lượng nội dung để ăn điểm constraint bề mặt;
- ở giai đoạn RL có thể rơi vào reward hacking hoặc output thoái hóa.

Project này nghiên cứu một pipeline hậu huấn luyện trên một GPU, kết hợp LoRA-SFT và custom GRPO, nhằm cải thiện đồng thời chất lượng tóm tắt và khả năng điều khiển đầu ra.

## 2. Định nghĩa tác vụ

Với source document hoặc document cluster `x`, instruction `c` và reference summary `y*`, mô hình sinh `ŷ` sao cho:

1. `ŷ` giữ được thông tin chính của `x`;
2. `ŷ` bám yêu cầu độ dài;
3. `ŷ` bám yêu cầu số câu;
4. `ŷ` không bị lặp vô nghĩa, blob hoặc mất nội dung.

Các constraint trong pipeline hiện tại dùng ba mẫu:

- `khoảng X từ/câu`;
- `trong khoảng lo-hi từ/câu`;
- `không quá X từ/câu`.

Trong code hiện tại, “từ” là đơn vị whitespace-delimited token. Báo cáo cần gọi rõ đây là đơn vị kỹ thuật của benchmark hiện có, không phải word segmentation ngôn ngữ học chuẩn.

## 3. Câu hỏi nghiên cứu chính cho báo cáo này

- **RQ1:** So với pretrained model, SFT-only, GRPO-fresh v5 và SFT+GRPO v5 cải thiện chất lượng và constraint adherence đến mức nào?
- **RQ2:** Backbone `Base` và `Instruct` khác nhau ra sao trước và sau hậu huấn luyện?
- **RQ3:** Gói thay đổi từ `v3 -> v5` có giảm được reward hacking, zero-content output và degenerate output hay không?
- **RQ4:** Configuration bundle `v4` và `v5` tạo ra trade-off gì giữa quality và length control?

Phần sau chỉ để appendix, không phải RQ chính:

- sentence-loss ablation;
- exploratory multi-document comparison;
- external baseline comparison theo nghĩa contextual.

## 4. Phạm vi báo cáo

### Trong phạm vi claim chính

- Single-document summarization trên VietNews và WikiLingua.
- Ma trận `Base/Instruct × {pretrained, SFT, GRPO-fresh v5, SFT+GRPO v5}`.
- Failure analysis `v3 -> v5` cho reward-hacking mitigation package.
- `v4 -> v5` configuration-bundle ablation.

### Chỉ mang tính exploratory/contextual

- ViMs và VLSP.
- Bảng baseline ngoài project.
- Historical combined aggregate `N=3100`.

### Ngoài phạm vi kết quả chính

- Persona/style control.
- DPO hoặc preference-pair training.
- Demo web/API.
- Kết luận nhân quả riêng cho sentence reward.

## 5. Thách thức kỹ thuật đã xử lý

| Thách thức | Cách xử lý hiện tại |
|---|---|
| Dữ liệu tiếng Việt không đồng nhất | Loader riêng cho 4 nguồn, chuẩn hóa về source/reference |
| Reference VietNews quá ngắn | Lọc title dưới 10 whitespace token cho SFT |
| Small LLM tuân thủ instruction yếu | Dùng SFT trước GRPO |
| Reward hacking | Dùng multiplicative content gate và degenerate-output detector |
| Repetition penalty phá output | Tắt repetition penalty / no-repeat n-gram trong run chính |
| VRAM/compute hạn chế | LoRA, gradient checkpointing, auto batch calibration, bf16 |

## 6. Tiêu chí thành công của báo cáo hiện tại

Với báo cáo project hiện tại, “thành công” không đồng nghĩa với việc phải rerun sạch toàn bộ. Thay vào đó, báo cáo cần:

- mô tả đúng provenance của mọi số liệu;
- không gọi sai decoding hiện có là greedy;
- giới hạn generalization claim chính vào VietNews/WikiLingua;
- gắn cảnh báo validity cho ViMs/VLSP và baseline ngoài;
- diễn giải `v3 -> v5` là mitigation package ablation, không phải detector-only ablation;
- giữ sentence-loss ablation ở appendix với kết luận `inconclusive`.

Các nâng cấp như split sạch cho ViMs/VLSP, greedy/seeded rerun, ROUGE chuẩn hóa đầy đủ và human evaluation hoàn chỉnh vẫn rất có giá trị, nhưng được xem là hướng tăng độ mạnh của paper về sau chứ không phải điều kiện bắt buộc để hoàn tất báo cáo project hiện tại.

## 7. Trạng thái đạt mục tiêu

| Mục tiêu | Trạng thái |
|---|---|
| Pipeline dữ liệu -> SFT -> GRPO -> eval | Hoàn thành |
| Bằng chứng quality improvement trên single-document | Có |
| Bằng chứng length control trên single-document | Có |
| So sánh Base và Instruct | Có |
| Reward-hacking failure analysis `v3 -> v5` | Có thể báo cáo từ artefact đã lưu |
| V4–V5 bundle ablation | Có |
| Sentence reward ablation | Có nhưng chưa kết luận |
| Generalization multi-document | Chưa xác nhận do leakage |
| Human evaluation pilot | Chưa chấm, mới khóa protocol |
| Persona/style control | Không nằm trong kết quả chính |
