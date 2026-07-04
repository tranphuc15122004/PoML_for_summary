# Project Audit for Current Report

**Cập nhật:** 03/07/2026  
**Mục đích:** khóa provenance, phạm vi claim và cách dùng các artefact hiện có cho báo cáo project/paper.

> Đây là nguồn chuẩn ở mức “project audit”. Khi có mâu thuẫn giữa các tài liệu lịch sử, ưu tiên tài liệu này, sau đó đến `docs/report.md`.

## 1. Quyết định khóa báo cáo

- Không chạy lại huấn luyện hoặc full evaluation.
- Giữ nguyên các generation đã lưu và coi chúng là artefact chính thức của project.
- Không mô tả evaluation hiện tại là greedy.
- Cách mô tả đúng là: `one-shot low-temperature decoding` với `temperature=0.3`, `top_p=0.9`, `do_sample=True`, và seed không được lưu.
- `v5` là cấu hình chính của paper.
- `v4` chỉ được dùng như configuration-bundle ablation.
- ViMs/VLSP và baseline ngoài project chỉ mang tính exploratory/contextual.

## 2. Canonical artefacts

| Artefact | Run ID / file | Vai trò trong báo cáo |
|---|---|---|
| Main Qwen3 evaluation | `models/eval_results/20260622_103706` | Kết quả chính cho pretrained, SFT, GRPO-fresh v4/v5, SFT+GRPO v4/v5 |
| Canonical v3 evaluation | `models/eval_results/20260627_043351` | Failure-analysis cho reward-hacking mitigation package |
| No-sentence ablation | `models/eval_results/20260630_020126` | Appendix-only ablation, kết luận inconclusive |
| Test input | `data/test.jsonl` | Nguồn chuẩn để nối lại constraint metadata với generation đã lưu |
| Reward implementation | `src/SFT_GRPO/rewards.py` | Định nghĩa `R_acc`, `R_len`, `R_sent`, degenerate detector |
| Evaluator | `src/evaluation/evaluate.py` | Định nghĩa decoding và metric aggregation hiện tại |

## 3. Ma trận model được dùng trong paper

### Main matrix

Với mỗi family `Qwen3 Base` và `Qwen3 Instruct`, paper chỉ dùng 4 trạng thái:

1. Pretrained.
2. SFT-only.
3. GRPO-fresh v5.
4. SFT+GRPO v5.

### Ablation / historical matrix

- `v4`: configuration-bundle ablation.
- `v3`: negative/failure baseline cho reward-hacking mitigation package.
- No-sentence ablation: appendix only.

## 4. Frozen evaluation protocol

### 4.1 Decoding

- `max_new_tokens=256`
- `temperature=0.3`
- `top_p=0.9`
- `do_sample=True`
- Không có evaluation seed được lưu

### 4.2 Metric hiện có

- `ROUGE-2`
- `relative length error`
- `length distance`
- sentence exact match / tolerant hit / MAE

### 4.3 Metric caveats

- `ROUGE-2` có thể đi qua fallback unique-bigram nếu dependency ngoài không sẵn sàng.
- `BARTScore` của run chính là `NaN`; không dùng trong claim.
- Sentence metrics hiện là audit hậu kiểm từ generation đã lưu, không phải output nguyên thủy của evaluator gốc.

## 5. Dữ liệu và validity boundary

### 5.1 Main generalization claims

Chỉ dựa trên hai tập held-out single-document:

- VietNews
- WikiLingua

### 5.2 Exploratory only

- ViMs
- VLSP

Lý do:

- ViMs test trùng `240` mẫu train + `60` mẫu validation.
- VLSP test trùng `285` mẫu train + `15` mẫu validation.

### 5.3 Nhóm metric/tập hợp chỉ dùng để đối chiếu lịch sử

- Combined `N=3100` được giữ làm aggregate lịch sử của artefact.
- Không dùng combined này như bằng chứng duy nhất cho generalization.

## 6. Những claim nào được phép nói

### Claim mạnh, hỗ trợ tốt

- Pipeline `SFT -> GRPO` hoạt động cho controllable Vietnamese summarization.
- SFT là bước cải thiện lớn nhất trong toàn pipeline.
- GRPO hiệu quả hơn rõ khi khởi tạo từ checkpoint sau SFT thay vì từ pretrained.
- `Qwen3 Instruct + SFT + GRPO v5` là cấu hình chất lượng nội dung tốt nhất trong các artefact Qwen3 hiện có trên single-document.
- Reward-hacking mitigation package từ `v3 -> v5` làm giảm mạnh degenerate / zero-content failure ở các nhánh bị lỗi nặng.

### Claim có điều kiện

- Sentence control được cải thiện ở nhiều nhánh, nhưng tác động riêng của sentence reward chưa được xác nhận.
- `v5` tốt hơn `v4` trong một số nhánh, nhưng đây là khác biệt của cả configuration bundle.
- ViMs/VLSP chỉ cho tín hiệu exploratory, không cho phép claim generalization multi-document.

### Claim không được nói

- “Evaluation là greedy.”
- “Project vượt trội trực tiếp so với GPT-4o / Qwen3-14B / Llama-70B.”
- “Cải thiện `v3 -> v5` là do riêng degenerate detector.”
- “Một hyperparameter cụ thể trong `v5` là nguyên nhân chính.”

## 7. Reward-hacking mitigation package: audit v3–v5

### 7.1 Định nghĩa vận hành

Các số dưới đây được tính lại từ generation đã lưu bằng implementation hiện tại trong `src/SFT_GRPO/rewards.py`.
Đây là `post-hoc audit` trên output cuối, không phải `reward_acc` online mà GRPO đã tối ưu trong lúc train:

- `degenerate rate`: `_is_degenerate(generated)`.
- `zero-content rate`: `R_acc = 0`.
- `near-zero content rate`: `R_acc <= 0.01`.
- `constraint-only hacking rate`: `R_acc <= 0.01` nhưng `R_len >= 0.8` và `R_sent >= 0.8`.
- `mean content reward`: trung bình `R_acc`.

### 7.2 Kết quả tổng hợp

| Run | ROUGE-2 | LenErr% | Degenerate% | R_acc=0% | R_acc<=0.01% | Hack% | Mean R_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base fresh v3 | 0.1058 | 163.16 | 23.81 | 27.16 | 27.71 | 0.55 | 0.1854 |
| Base fresh v5 | 0.1866 | 29.05 | 0.32 | 1.94 | 1.97 | 0.84 | 0.3160 |
| Base SFT-init v3 | 0.1803 | 425.00 | 14.77 | 1.16 | 1.58 | 0.42 | 0.2508 |
| Base SFT-init v5 | 0.2638 | 14.94 | 2.71 | 0.45 | 0.45 | 0.42 | 0.4009 |
| Instruct fresh v3 | 0.1500 | 30.08 | 0.03 | 1.48 | 1.52 | 1.06 | 0.2768 |
| Instruct fresh v5 | 0.1669 | 22.78 | 0.03 | 1.58 | 1.61 | 1.32 | 0.2979 |
| Instruct SFT-init v3 | 0.2506 | 19.94 | 0.94 | 1.19 | 1.19 | 0.90 | 0.3837 |
| Instruct SFT-init v5 | 0.2765 | 15.85 | 1.42 | 0.45 | 0.48 | 0.39 | 0.4213 |

### 7.3 Diễn giải đúng mức

- `train_metrics.csv` của bốn run `v3` cho thấy `reward_acc` online thực sự gần 0 trong suốt quá trình train: mean lần lượt là `0.0002`, `0.0019`, `0.0009`, và `0.0036` cho `Base fresh`, `Base SFT-init`, `Instruct fresh`, `Instruct SFT-init`. Vì vậy bảng trên không được đọc như “training `R_acc` của v3 vẫn cao”; nó là audit lại trên generations cuối bằng reward code hiện tại.
- Cặp bị lỗi nặng nhất là `Base fresh`: `degenerate rate` giảm từ `23.81%` xuống `0.32%`, đồng thời `mean R_acc` tăng từ `0.1854` lên `0.3160`.
- `Base SFT-init v3` cũng bị lỗi nội dung và độ dài rất nặng; `v5` kéo `LenErr` từ `425.00%` xuống `14.94%`.
- `Hack%` không nên đọc một mình. Nhiều lỗi `v3` là repetition loop hoặc punctuation flood, nên failure mode lộ ra rõ hơn khi xem cùng `Degenerate%`, `R_acc=0%`, `R_acc<=0.01%` và các paired outputs phía dưới.
- Hai nhánh `Instruct` vốn đã ổn định hơn ở `v3`, nên `v3 -> v5` chủ yếu cho thấy cải thiện chất lượng/độ dài chứ không phải “sửa sập hoàn toàn”.
- Vì `v3` và `v5` khác nhau ở nhiều thay đổi hệ thống, đây phải được gọi là ablation của `reward-hacking mitigation package`, không phải detector-only ablation.

### 7.4 Ví dụ paired outputs

1. `Base fresh`, chỉ số `8`, VietNews  
   Reference: `Chuyện chưa biết về vụ đánh án sới bạc nghìn đô trên núi Trạng Nẹo`  
   v3: câu dài, lệch nội dung, mô tả chung chung về lực lượng cảnh giới  
   v5: `Đường dây đánh bạc nghìn đô trang bị bộ đàm cảnh giới.`

2. `Base fresh`, chỉ số `14`, VietNews  
   Reference: `Điều tra vụ anh rể dùng dao đâm chết em vợ vào rạng sáng`  
   v3: lặp `assistant` liên tiếp, gần như rỗng nội dung  
   v5: `Anh C. bị anh rể đâm chết tại nhà.`

3. `Base SFT-init`, chỉ số `7`, VietNews  
   Reference: `5 năm chưa xử xong vụ án “ lạ ” có hơn 30 luật sư bào chữa miễn phí`  
   v3: sinh đoạn dài lặp dấu chấm, length blow-up rất rõ  
   v5: `Vụ án “ lạ ” : Huỷ án 2 lần , bị cáo vẫn bị tạm giam 5 năm !`

4. `Instruct SFT-init`, chỉ số `130`, VietNews  
   Reference: `Công an chính thức thông tin diễn biến vụ tài xế Grab bị cứa cổ ở Thủ Đức`  
   v3: chỉ giữ lại một phần diễn biến cướp xe  
   v5: `Bắt nam thanh niên cứa cổ tài xế Grab cướp xe máy ở TP. HCM`

## 8. V4–V5 configuration-bundle ablation

### 8.1 Bundle definition

- `v4`: `K=4`, `LR=5e-7`, `beta=0.15`
- `v5`: `K=8`, `LR=2e-6`, `beta=0.04`

### 8.2 Nguyên tắc diễn giải

- Không quy công cho một hyperparameter riêng lẻ.
- Chỉ kết luận về độ nhạy của cả configuration bundle.

### 8.3 Kết luận gọn

- `v5` giúp các nhánh `fresh` tốt hơn rõ rệt.
- Ở nhánh `Base SFT-init`, `v5` giữ chất lượng gần `v4` nhưng siết length tốt hơn.
- Ở nhánh `Instruct SFT-init`, `v5` tăng ROUGE-2 nhưng không luôn cho length control tốt hơn.

## 9. Sentence-loss ablation

- Nguồn: `models/eval_results/20260630_020126`
- Chỉ có `Base fresh` và `Base SFT-init`
- Evaluation vẫn là sampling không lưu seed
- Kết luận hợp lệ: `inconclusive`

Vì vậy:

- không đưa vào main paper body như contribution chính;
- chỉ giữ ở appendix như một ablation chưa đủ sạch để kết luận nhân quả.

## 10. Pilot human evaluation specification

Nếu kịp triển khai trước khi khóa LaTeX:

- Chọn `40` mẫu với seed `42`
- `15` VietNews, `15` WikiLingua, `5` ViMs, `5` VLSP
- Chấm ba output trong họ Instruct: pretrained, SFT-only, SFT+GRPO v5
- Tổng cộng `120` output
- Rubric `1–5`: factuality, fluency, instruction adherence
- Một người chấm duy nhất; báo cáo như `pilot human evaluation`

Nếu không hoàn thành kịp:

- chuyển thành future work / protocol appendix;
- không để thiếu phần này làm thay đổi narrative chính của paper.

## 11. Tài liệu nào nên dùng khi viết

- `docs/PROJECT_AUDIT.md`: provenance, validity, claim boundary
- `docs/report.md`: bảng kết quả và diễn giải chính
- `docs/REPORT_OUTLINE.md`: khung ACL 8 trang
- `docs/qwen3_training_report.md`: snapshot lịch sử, không phải nguồn claim cuối
