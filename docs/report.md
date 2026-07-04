# Báo cáo kết quả thực nghiệm hiện tại

**Main run:** `models/eval_results/20260622_103706`  
**Reward-hacking comparison:** `models/eval_results/20260627_043351`  
**Sentence-loss ablation:** `models/eval_results/20260630_020126`

> Báo cáo này dùng đúng các artefact đã lưu. Không có rerun mới. Evaluation hiện tại phải được mô tả là `one-shot low-temperature decoding` với `temperature=0.3`, `top_p=0.9`, `do_sample=True`, và seed không được lưu.

## 1. Quy ước đọc kết quả

- Kết luận generalization chính chỉ dựa trên VietNews và WikiLingua.
- ViMs/VLSP vẫn có mặt trong aggregate lịch sử nhưng phải gắn cảnh báo leakage.
- `v5` là cấu hình chính của paper.
- `v4` chỉ dùng như configuration-bundle ablation.
- `v3` là failure baseline cho reward-hacking mitigation package.
- Sentence metrics dưới đây được tính lại từ generation đã lưu theo cùng heuristic đếm câu trên tất cả model trong bảng chính.

## 2. Ma trận model chính của paper

| Family | Pretrained | SFT-only | GRPO-fresh v5 | SFT+GRPO v5 |
|---|---|---|---|---|
| Base | `QWEN3_BASE_base` | `QWEN3_BASE_sft` | `QWEN3_BASE_grpo_fresh_v5` | `QWEN3_BASE_grpo_sft_v5` |
| Instruct | `QWEN3_INSTRUCT_base` | `QWEN3_INSTRUCT_sft` | `QWEN3_INSTRUCT_grpo_fresh_v5` | `QWEN3_INSTRUCT_grpo_sft_v5` |

## 3. Kết quả held-out single-document

### 3.1 ROUGE-2

| Model | VietNews | WikiLingua |
|---|---:|---:|
| Base pretrained | 0.1213 | 0.0318 |
| Base + SFT | 0.2661 | 0.2443 |
| Base + fresh GRPO v5 | 0.2073 | 0.0971 |
| Base + SFT + GRPO v5 | 0.2897 | **0.2635** |
| Instruct pretrained | 0.1648 | 0.0744 |
| Instruct + SFT | 0.2670 | 0.2393 |
| Instruct + fresh GRPO v5 | 0.1867 | 0.0809 |
| Instruct + SFT + GRPO v5 | **0.2962** | 0.2559 |

### 3.2 Relative length error (%)

| Model | VietNews | WikiLingua |
|---|---:|---:|
| Base pretrained | 205.89 | 174.67 |
| Base + SFT | 31.19 | 24.72 |
| Base + fresh GRPO v5 | 22.36 | 50.94 |
| Base + SFT + GRPO v5 | 6.43 | **14.69** |
| Instruct pretrained | 24.50 | 57.47 |
| Instruct + SFT | **6.26** | 18.34 |
| Instruct + fresh GRPO v5 | 17.08 | 38.67 |
| Instruct + SFT + GRPO v5 | 9.24 | 18.11 |

### 3.3 Length distance

| Model | VietNews | WikiLingua |
|---|---:|---:|
| Base pretrained | 33.0 | 61.6 |
| Base + SFT | 5.73 | 14.18 |
| Base + fresh GRPO v5 | 3.8 | 22.0 |
| Base + SFT + GRPO v5 | 1.16 | **8.35** |
| Instruct pretrained | 4.0 | 23.4 |
| Instruct + SFT | 1.09 | 9.90 |
| Instruct + fresh GRPO v5 | 2.9 | 16.1 |
| Instruct + SFT + GRPO v5 | 1.52 | 10.25 |

### 3.4 Sentence control trên VietNews

| Model | Exact match ↑ | Tolerant hit ±1 ↑ | MAE ↓ |
|---|---:|---:|---:|
| Base pretrained | 73.95% | 89.55% | 0.550 |
| Base + SFT | 91.40% | 96.55% | 4.299 |
| Base + fresh GRPO v5 | 91.80% | 99.30% | 0.089 |
| Base + SFT + GRPO v5 | 93.85% | 98.75% | 0.316 |
| Instruct pretrained | 93.95% | 99.40% | 0.067 |
| Instruct + SFT | 95.45% | 98.90% | 0.058 |
| Instruct + fresh GRPO v5 | 94.85% | **99.60%** | 0.056 |
| Instruct + SFT + GRPO v5 | **96.95%** | 99.05% | **0.041** |

### 3.5 Sentence control trên WikiLingua

| Model | Exact match ↑ | Tolerant hit ±1 ↑ | MAE ↓ |
|---|---:|---:|---:|
| Base pretrained | 15.4% | 32.4% | 2.910 |
| Base + SFT | 58.6% | 83.8% | 1.674 |
| Base + fresh GRPO v5 | 34.2% | 72.8% | 1.226 |
| Base + SFT + GRPO v5 | 65.6% | 95.6% | 0.470 |
| Instruct pretrained | 34.2% | 64.6% | 1.582 |
| Instruct + SFT | **88.0%** | **97.0%** | **0.204** |
| Instruct + fresh GRPO v5 | 46.8% | 72.4% | 1.178 |
| Instruct + SFT + GRPO v5 | 78.8% | 95.6% | 0.340 |

### 3.6 Kết luận chính từ main matrix

1. SFT là bước cải thiện lớn nhất ở cả hai family.
2. GRPO-fresh giúp hơn pretrained, nhưng vẫn kém rõ so với SFT-only và SFT+GRPO.
3. `Instruct + SFT + GRPO v5` là cấu hình mạnh nhất về quality trên VietNews và toàn bộ artefact hiện có.
4. Về length control, `Base + SFT + GRPO v5` mạnh nhất trong họ Base; còn ở họ Instruct, `SFT-only` vẫn cạnh tranh hoặc tốt hơn `SFT+GRPO v5` trên một số metric độ dài.

## 4. Aggregate lịch sử trên full test `N=3100`

> Bảng này hữu ích để đối chiếu artefact và theo dõi toàn run, nhưng không được dùng một mình để claim generalization vì ViMs/VLSP bị leakage.

| Model | ROUGE-2 | LenErr% |
|---|---:|---:|
| Base pretrained | 0.1056 | 169.77 |
| Base + SFT | 0.2632 | 30.52 |
| Base + fresh GRPO v5 | 0.1866 | 29.05 |
| Base + SFT + GRPO v5 | 0.2638 | 14.94 |
| Instruct pretrained | 0.1506 | 30.22 |
| Instruct + SFT | 0.2609 | **12.42** |
| Instruct + fresh GRPO v5 | 0.1669 | 22.78 |
| Instruct + SFT + GRPO v5 | **0.2765** | 15.85 |

## 5. Reward-hacking mitigation package: `v3 -> v5`

### 5.1 Định nghĩa audit

Các số dưới đây được tính lại từ generation đã lưu bằng implementation hiện tại trong `src/SFT_GRPO/rewards.py`.
Đây là `post-hoc audit` trên output cuối, không phải `reward_acc` online mà GRPO đã thấy trong lúc train:

- `degenerate rate`: `_is_degenerate(generated)`.
- `zero-content rate`: `R_acc = 0`.
- `near-zero content rate`: `R_acc <= 0.01`.
- `constraint-only hacking rate`: `R_acc <= 0.01` nhưng `R_len >= 0.8` và `R_sent >= 0.8`.
- `mean content reward`: trung bình `R_acc`.

### 5.2 Kết quả tổng hợp

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

### 5.3 Cách diễn giải đúng mức

- `train_metrics.csv` của bốn run `v3` đúng là cho thấy `reward_acc` gần 0 trong suốt quá trình train: mean lần lượt chỉ là `0.0002` (Base fresh), `0.0019` (Base SFT-init), `0.0009` (Instruct fresh), và `0.0036` (Instruct SFT-init). Vì vậy bảng trên không được đọc như “training `R_acc` của v3 vẫn cao”; nó là audit lại trên generations cuối bằng reward code hiện tại.
- `Base fresh` là failure case rõ nhất: `degenerate rate` giảm từ `23.81%` xuống `0.32%`, còn `mean R_acc` tăng từ `0.1854` lên `0.3160`.
- `Base SFT-init v3` không sập hoàn toàn như `Base fresh v3`, nhưng bị lỗi độ dài rất nặng; `v5` kéo `LenErr` từ `425.00%` xuống `14.94%`.
- `Hack%` không nên đọc riêng lẻ. Nhiều lỗi `v3` là repetition loop hoặc punctuation flood, tức là vừa hỏng nội dung vừa hỏng luôn constraint; vì thế reward hacking hiện ra rõ hơn khi đọc cùng `Degenerate%`, `R_acc=0%`, `R_acc<=0.01%` và các paired outputs bên dưới.
- Hai nhánh `Instruct` vốn đã ổn định hơn ở `v3`, nên `v3 -> v5` chủ yếu là cải thiện chất lượng và độ dài chứ không phải “sửa từ hỏng thành dùng được”.
- Vì `v3` và `v5` khác nhau ở nhiều thay đổi hệ thống, phần này phải được gọi là ablation của `reward-hacking mitigation package`, không phải detector-only ablation.

### 5.4 Ví dụ paired outputs

1. `Base fresh`, chỉ số `14`, VietNews  
   Reference: `Điều tra vụ anh rể dùng dao đâm chết em vợ vào rạng sáng`  
   v3: `assistant` lặp 128 token, `ROUGE-2 = 0.0`, `LenErr = 814.29%`  
   v5: `Anh C. bị anh rể đâm chết tại nhà.`
2. `Base SFT-init`, chỉ số `7`, VietNews  
   Reference: `5 năm chưa xử xong vụ án “ lạ ” có hơn 30 luật sư bào chữa miễn phí`  
   v3: mở đầu còn bám source nhưng sau đó blow-up thành chuỗi dấu chấm, dài 245 từ, `LenErr = 1189.47%`  
   v5: `Vụ án “ lạ ” : Huỷ án 2 lần , bị cáo vẫn bị tạm giam 5 năm !`
3. `Instruct SFT-init`, chỉ số `130`, VietNews: v3 chỉ giữ lại một phần diễn biến; v5 sinh được câu đúng tâm điểm `Bắt nam thanh niên cứa cổ tài xế Grab cướp xe máy ở TP. HCM`.

## 6. V4–V5 configuration-bundle ablation

> Diễn giải ở mức bundle, không quy công cho từng hyperparameter riêng lẻ.

### 6.1 Bundle definition

- `v4`: `K=4`, `LR=5e-7`, `beta=0.15`
- `v5`: `K=8`, `LR=2e-6`, `beta=0.04`

### 6.2 Macro average trên hai tập held-out chính

| Branch | v4 ROUGE-2 | v5 ROUGE-2 | v4 LenErr% | v5 LenErr% | v4 LenDist | v5 LenDist |
|---|---:|---:|---:|---:|---:|---:|
| Base fresh | 0.1093 | 0.1522 | 91.67 | 36.65 | 26.70 | 12.90 |
| Base SFT-init | 0.2625 | 0.2766 | 17.88 | 10.56 | 7.05 | 4.80 |
| Instruct fresh | 0.1239 | 0.1338 | 33.98 | 27.88 | 11.65 | 9.50 |
| Instruct SFT-init | 0.2624 | 0.2761 | 13.09 | 13.68 | 6.00 | 5.85 |

### 6.3 Kết luận

- `v5` thắng rõ ở tất cả nhánh `fresh`.
- `v5` cũng cải thiện nhánh `Base SFT-init` trên cả quality và length control ở phần held-out single-document.
- Với `Instruct SFT-init`, `v5` tăng quality nhưng length error không còn tốt hơn `v4`; đây là trade-off quan trọng nhất của bundle ablation.

## 7. Sentence-loss ablation

> Phần này chỉ nên để ở appendix.

Nguồn: `models/eval_results/20260630_020126`

| Branch | w_sent | ROUGE-2 | LenErr% | Sent exact | Sent MAE |
|---|---:|---:|---:|---:|---:|
| Base fresh v4 | 0.2 | 0.1388 | 76.44 | 59.42% | 1.136 |
| Base fresh ablation | 0.0 | 0.1399 | 76.50 | 60.32% | 1.163 |
| Base SFT-init v4 | 0.2 | 0.2671 | 20.82 | 73.39% | 2.009 |
| Base SFT-init ablation | 0.0 | 0.2685 | 17.21 | 73.65% | 1.452 |

Kết luận hợp lệ ở thời điểm này: `inconclusive`. Hai phía vẫn được evaluate bằng sampling không lưu seed, nên chưa thể quy vai trò nhân quả riêng cho sentence reward.

## 8. Exploratory multi-document và baseline ngoài

### 8.1 Multi-document

- ViMs test trùng `240` train + `60` val.
- VLSP test trùng `285` train + `15` val.
- Vì vậy mọi bảng ViMs/VLSP chỉ nên để ở appendix hoặc ghi rõ `exploratory`.

### 8.2 Baseline ngoài project

`VDT_Textsum/ketqua.md` có thể dùng để đặt bối cảnh tham khảo cho các model như VietAI, Qwen3-14B, Qwen3-4B, Llama3.3-70B, GPT-4o. Tuy nhiên:

- thiếu protocol đồng nhất về split, prompt, decoding và metric implementation;
- không dùng để tuyên bố project “vượt” các model đó;
- trong main paper chỉ nên so sánh theo `ROUGE-2` và `Length Distance`, còn bảng đầy đủ chuyển sang appendix.

## 9. Claim chốt cho paper

1. Project xây dựng thành công pipeline `SFT -> GRPO` cho controllable Vietnamese summarization.
2. SFT là bước cải thiện lớn nhất, còn GRPO hiệu quả rõ nhất khi khởi tạo từ checkpoint sau SFT.
3. `Qwen3 Instruct + SFT + GRPO v5` là cấu hình nội dung mạnh nhất trong các artefact chính hiện có.
4. `v3 -> v5` cho thấy reward-hacking mitigation package làm giảm mạnh failure mode ở các nhánh Base, đặc biệt `Base fresh`.
5. Phần multi-document, baseline ngoài và sentence-loss ablation phải được trình bày với caveat rõ ràng, không đẩy thành claim mạnh vượt quá evidence.
