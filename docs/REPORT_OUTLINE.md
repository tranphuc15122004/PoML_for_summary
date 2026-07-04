# ACL-Style Report Outline

**Mục tiêu:** chuyển kết quả hiện có thành một báo cáo project theo phong cách ACL, ưu tiên trung thực về protocol và validity hơn là mở rộng thêm thực nghiệm.

## 1. Framing chung

### Core story

- Hậu huấn luyện Qwen3-4B cho tóm tắt tiếng Việt có kiểm soát.
- So sánh `Base` và `Instruct`.
- So sánh `pretrained`, `SFT-only`, `GRPO-fresh v5`, `SFT+GRPO v5`.
- Phân tích `reward-hacking mitigation package` qua `v3 -> v5`.
- Dùng `v4` như configuration-bundle ablation.

### Narrative discipline

- Không nói evaluation là greedy.
- Không dùng ViMs/VLSP để claim generalization chính.
- Không dùng baseline ngoài để claim vượt trội trực tiếp.
- Sentence-loss ablation chỉ ở appendix.

## 2. Khung 8 trang

### 1. Introduction

Mục tiêu:

- đặt bài toán controllable Vietnamese summarization;
- nêu khó khăn của small/open LLM với độ dài, số câu và reward hacking;
- chốt 3 contribution chính.

Contribution nên viết:

1. Xây dựng pipeline dữ liệu -> LoRA-SFT -> custom GRPO -> evaluation cho tiếng Việt.
2. Cung cấp ma trận thực nghiệm `Base/Instruct × 4 trạng thái hậu huấn luyện`.
3. Phân tích failure mode `v3 -> v5` cho reward-hacking mitigation package.

### 2. Task and Problem Setting

Nội dung:

- định nghĩa input/source + constraint;
- ba loại constraint về độ dài/số câu;
- đơn vị đếm là whitespace-delimited tokens;
- main claim chỉ trên single-document held-out.

Nguồn chính:

- `docs/problem_statement.md`
- `docs/DATASETS.md`

### 3. Data and Validity

Nội dung:

- bốn dataset: VietNews, WikiLingua, ViMs, VLSP;
- quy mô split sinh ra;
- leakage audit;
- vì sao ViMs/VLSP chỉ exploratory.

Table đề xuất:

- `Table 1`: nguồn dữ liệu, số mẫu train/val/test, loại tác vụ, validity tag.

### 4. Method

Nội dung:

- prompt/data format cho SFT và GRPO;
- reward hiện hành:
  - `R_acc = 0.5 * ROUGE-1 + 0.5 * ROUGE-L`
  - `R_total = R_acc * (1 + w_len * R_len + w_sent * R_sent)`
- degenerate detector;
- khác biệt giữa `GRPO-fresh` và `GRPO SFT-init`.

Figure đề xuất:

- `Figure 1`: pipeline tổng quát từ dataset đến evaluation.
- `Figure 2`: sơ đồ reward gate và failure mode reward hacking.

### 5. Experimental Setup

Nội dung:

- model families: Qwen3 Base, Qwen3 Instruct;
- main evaluation run `20260622_103706`;
- `v3` canonical run `20260627_043351`;
- decoding protocol trung thực;
- metric set dùng trong report;
- no-sentence ablation để appendix.

Table đề xuất:

- `Table 2`: model matrix và run ID dùng trong paper.

### 6. Main Results

Đây là section quan trọng nhất.

#### 6.1 Main matrix

Chỉ trình bày:

- pretrained
- SFT-only
- GRPO-fresh v5
- SFT+GRPO v5

Table đề xuất:

- `Table 3`: ROUGE-2 trên VietNews/WikiLingua cho 8 model chính.
- `Table 4`: Relative length error và length distance trên VietNews/WikiLingua.
- `Table 5`: Sentence exact / tolerant hit / MAE trên VietNews/WikiLingua.

Thông điệp cần rút ra:

- SFT là bước cải thiện lớn nhất.
- GRPO cần SFT-init để phát huy rõ.
- Instruct + SFT + GRPO v5 là cấu hình chất lượng tốt nhất.

#### 6.2 Reward-hacking mitigation package

Table đề xuất:

- `Table 6`: `v3 -> v5` cho bốn cặp tương ứng với các cột:
  - ROUGE-2
  - LenErr%
  - degenerate rate
  - zero-content rate
  - near-zero-content rate
  - constraint-only hacking rate
  - mean `R_acc`

Thông điệp:

- Base fresh là failure case rõ nhất.
- `v3 -> v5` làm giảm mạnh degenerate outputs.
- Đây là package-level ablation, không phải detector-only ablation.

#### 6.3 V4–V5 configuration bundle ablation

Table đề xuất:

- `Table 7`: so sánh `v4` và `v5` cho 4 nhánh `fresh/SFT-init × Base/Instruct`.

Thông điệp:

- `v5` hữu ích nhất ở các nhánh `fresh`;
- với nhánh mạnh sẵn, `v4` và `v5` tạo trade-off khác nhau giữa content và length.

### 7. Qualitative and Human Evaluation

Nếu hoàn thành kịp:

- `Table 8` hoặc figure phụ cho pilot human evaluation 40 mẫu.
- 4–6 ví dụ paired `v3` vs `v5`.

Nếu không hoàn thành kịp:

- giữ 4–6 paired examples trong main text;
- chuyển human evaluation protocol sang appendix/future work.

### 8. Limitations and Conclusion

Limitations bắt buộc nêu:

- decoding là low-temperature sampling, không có seed lưu lại;
- ViMs/VLSP bị leakage;
- baseline ngoài project không cùng protocol;
- sentence-loss ablation chưa kết luận;
- SFT run dùng TRL 0.13 nên không có assistant-only masking như kỳ vọng ban đầu.

Conclusion cần ngắn và kỷ luật:

- project thành công ở mức pipeline + empirical study;
- claim được giới hạn đúng theo evidence đang có.

## 3. Bảng và hình nên có

### Main body

1. `Table 1`: Datasets, split sizes, validity tags.
2. `Table 2`: Model matrix và run IDs.
3. `Table 3`: Main ROUGE-2 results trên VietNews/WikiLingua.
4. `Table 4`: Main length results trên VietNews/WikiLingua.
5. `Table 5`: Main sentence-control results trên VietNews/WikiLingua.
6. `Table 6`: Reward-hacking mitigation package (`v3 -> v5`).
7. `Table 7`: `v4 -> v5` configuration-bundle ablation.
8. `Figure 1`: pipeline.
9. `Figure 2`: reward gate / failure mode diagram.

### Appendix

- Full per-dataset table gồm cả ViMs/VLSP.
- Historical combined aggregate.
- Sentence-loss ablation.
- External baseline full table.
- Additional qualitative examples.

## 4. Appendix plan

### Appendix A

Full per-dataset results cho tất cả Qwen3 variants trong run `20260622_103706`.

### Appendix B

No-sentence ablation:

- Base fresh
- Base SFT-init
- kết luận `inconclusive`

### Appendix C

Exploratory multi-document:

- ViMs
- VLSP
- luôn gắn leakage warning

### Appendix D

External baselines:

- VietAI
- Qwen3-14B
- Qwen3-4B
- Llama3.3-70B
- GPT-4o

Chỉ dùng làm contextual comparison.

### Appendix E

Paired qualitative outputs `v3 -> v5`.

## 5. Việc còn lại trước khi chuyển sang LaTeX

- Đồng bộ các Markdown nguồn: `problem_statement`, `report`, `DATASETS`, `pipeline_plan`, `progress_report`.
- Chọn 4–6 paired examples cuối cùng cho main text.
- Nếu kịp, thực hiện pilot human evaluation 40 mẫu.
- Chuyển bảng/hình theo đúng giới hạn 8 trang ACL.
