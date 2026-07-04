# Báo cáo huấn luyện Qwen3-4B cho bài toán tóm tắt tiếng Việt có ràng buộc

> **Trạng thái tài liệu:** đây là snapshot lịch sử của đợt đánh giá 22/06/2026. Khi viết báo cáo cuối, dùng `docs/PROJECT_AUDIT.md`, `docs/report.md` và `docs/REPORT_OUTLINE.md` làm nguồn chuẩn. Narrative hiện đã khóa như sau: `v5` là cấu hình chính, `v4` là configuration-bundle ablation, `v3` là reward-hacking mitigation package baseline, và no-sentence ablation chỉ ở appendix. Audit sau đó phát hiện: (1) ViMs/VLSP test overlap hoàn toàn với train/validation; (2) evaluation sampling không cố định seed; (3) SFT trên TRL 0.13 không dùng assistant-only loss; và (4) bảng baseline ngoài project không cùng protocol nên không hỗ trợ tuyên bố xếp hạng trực tiếp.

<div align="center">

**Mã đánh giá:** `20260622_103706`  
**Phạm vi:** chỉ phân tích các biến thể `Qwen3-4B`

</div>

---

## 1. Mục tiêu của báo cáo này

Báo cáo tập trung trả lời ba câu hỏi:

1. Quá trình huấn luyện đã diễn ra như thế nào
2. Mô hình đã đáp ứng yêu cầu của bài toán đến mức nào
3. Mỗi thành phần post-training tác động ra sao lên chất lượng tóm tắt và khả năng kiểm soát độ dài

> **Điểm cần khóa ngay từ đầu:**
> - Hệ thống hiện tại có bằng chứng đánh giá cuối cho `quality` và `length control`
> - Pipeline có huấn luyện/reward liên quan đến `sentence control`, nhưng run đánh giá này **không có metric tổng kết riêng** cho mức độ tuân thủ số câu
> - Ý tưởng `persona/style control` chưa được phản ánh thành kết quả thực nghiệm hoàn chỉnh, nên **không được coi là mục tiêu đã đạt**

## 2. Dữ liệu, mô hình và metric đang thực sự được dùng

### 2.1 Dữ liệu

Các file hiện có tại thời điểm viết báo cáo:

| File | Số mẫu |
|---|---:|
| `data/sft_train.jsonl` | 111,150 |
| `data/sft_val.jsonl` | 2,500 |
| `data/grpo_train.jsonl` | 119,942 |
| `data/grpo_val.jsonl` | 21,897 |
| `data/test.jsonl` | 3,100 |

#### Nguồn gốc dữ liệu

Dữ liệu được lấy từ bốn tập thuộc `VDT_Textsum/`:

| Dataset | Loại | Số lượng raw | Vai trò |
|---|---|---|---|
| **VietNews** | single-document, abstractive | ~150K news articles (title = summary) | Nguồn chính cho cả SFT và GRPO |
| **WikiLingua** | single-document, abstractive | ~14K wiki articles | Bổ sung chất lượng cao cho cả SFT và GRPO |
| **VLSP 2022 AbMuSu** | multi-document | ~285 train / 15 val | Loader ưu tiên trường `summary`, fallback extractive label; chỉ dùng cho GRPO |
| **ViMs** | multi-document, abstractive | 300 clusters, 1 gold summary mỗi cluster | Bổ sung multi-doc cho cả SFT và GRPO |

#### Cấu trúc từng tập huấn luyện

**SFT train (111,150 mẫu):**

| Nguồn | Số mẫu | Tỷ lệ |
|---|---:|---:|
| VietNews (title ≥ 10 từ) | 96,911 | 87.2% |
| WikiLingua | 13,999 | 12.6% |
| ViMs (80% clusters) | 240 | 0.2% |
| **Tổng** | **111,150** | **100%** |

**GRPO train (119,942 mẫu):**

| Nguồn | Số mẫu | Tỷ lệ |
|---|---:|---:|
| VietNews (toàn bộ) | 105,418 | 87.9% |
| WikiLingua | 13,999 | 11.7% |
| VLSP | 285 | 0.2% |
| ViMs (80% clusters) | 240 | 0.2% |
| **Tổng** | **119,942** | **100%** |

#### Tập validation

| File | Số mẫu | Nguồn gốc |
|---|---:|---|
| `data/sft_val.jsonl` | 2,500 | 2,000 VietNews val + 500 WikiLingua val |
| `data/grpo_val.jsonl` | 21,897 | VietNews val còn lại (~20,642) + WikiLingua val còn lại (~1,180) + VLSP val (15) + ViMs 20% (60) |

> **Tại sao SFT val chỉ có VietNews và WikiLingua, không có ViMs và VLSP?**
>
> - **ViMs**: Bộ này chỉ có 300 cụm, được chia 80/20: 240 cụm vào SFT train, 60 cụm còn lại vào GRPO val. 60 mẫu quá nhỏ để tạo tập validation ổn định cho SFT. ViMs là multi-document (chỉ chiếm 0.2% SFT train), nên không đại diện cho phần dữ liệu chính. 2,500 mẫu single-document từ VietNews + WikiLingua là đủ để theo dõi loss và phát hiện overfitting.
>
> - **VLSP**: Không xuất hiện trong SFT. Loader hiện ưu tiên human-written `summary` và chỉ fallback sang extractive label; quyết định GRPO-only chủ yếu do quy mô nhỏ và thiết kế pipeline lịch sử.

#### Tập test

Tập test 3,100 mẫu gồm:

| Dataset | Số mẫu | Loại |
|---|---:|---:|
| VietNews | 2,000 | single-document |
| WikiLingua | 500 | single-document |
| ViMs | 300 | multi-document |
| VLSP | 300 | multi-document |

#### Tại sao `sft_train` (111,150) nhỏ hơn `grpo_train` (119,942)?

Chênh lệch **8,792 mẫu** đến từ hai nguyên nhân:

1. **Lọc VietNews theo độ dài title (nguyên nhân chính — ~8,507 mẫu):**  
   SFT yêu cầu chất lượng target cao nên chỉ giữ các mẫu VietNews có title ≥ 10 từ (`VIETNEWS_MIN_TARGET_WORDS` trong `augmenter.py`). Các title quá ngắn (thường là click-bait hoặc không đủ thông tin để học) bị loại khỏi SFT. GRPO không áp dụng bộ lọc này vì reward function có thể xử lý được các title ngắn trong quá trình alignment.

2. **VLSP chỉ dùng cho GRPO (285 mẫu):**  
   Loader ưu tiên trường `summary` và fallback sang label-based extraction. Việc không đưa VLSP vào SFT là lựa chọn của pipeline hiện tại, không nên diễn giải rằng toàn bộ target VLSP đều là extractive.

Tóm lại: `sft_train` nhỏ hơn vì tiêu chí chọn dữ liệu cho SFT khắt khe hơn — ưu tiên chất lượng target, trong khi GRPO chấp nhận dùng nhiều nguồn hơn và để reward điều chỉnh hành vi.

---

### 2.2 12 biến thể Qwen3-4B được so sánh

| Họ mô hình | Các biến thể |
|---|---|
| `QWEN3_BASE_*` | `base`, `sft`, `grpo_fresh_v4`, `grpo_fresh_v5`, `grpo_sft_v4`, `grpo_sft_v5` |
| `QWEN3_INSTRUCT_*` | `base`, `sft`, `grpo_fresh_v4`, `grpo_fresh_v5`, `grpo_sft_v4`, `grpo_sft_v5` |

Trong đó:

- `fresh`: GRPO khởi tạo trực tiếp từ pretrained model.
- `sft`: GRPO khởi tạo từ checkpoint sau SFT.
- `v4` và `v5`: hai cấu hình GRPO khác nhau theo cụm tham số, không phải thay đổi một tham số đơn lẻ.

Cụ thể, các run được báo cáo ở đây dùng hai bundle hyperparameter sau:

| Biến thể | `num_generations` (K) | `learning_rate` | `beta` |
|---|---:|---:|---:|
| `v4` | 4 | `5e-7` | `0.15` |
| `v5` | 8 | `2e-6` | `0.04` |

Các tham số GRPO quan trọng khác vẫn được giữ nguyên giữa `v4` và `v5`:

| Tham số chung | Giá trị | Ghi chú |
|---|---:|---|
| `temperature` | `0.7` | nhiệt độ sinh rollout |
| `top_p` | `0.9` | top-p sampling |
| `max_new_tokens` | `80` | giới hạn token sinh mỗi completion |
| `max_seq_length` | `3072` | H200 dùng 3072; A100-40G thường hạ xuống 2048 |
| `total_steps` | `800` | số bước GRPO của run |
| `reward_weight_accuracy` | `0.5` | trọng số `R_acc` |
| `reward_weight_length` | `0.3` | trọng số `R_len` |
| `reward_weight_sentence` | `0.2` | trọng số `R_sent` |
| `length_advantage_alpha` | `0.0` | tắt length-scaled advantage |
| `repetition_penalty` | `1.0` | tắt penalty lặp |
| `no_repeat_ngram_size` | `0` | tắt chặn n-gram lặp |
| `bf16/fp16` | theo GPU | `bf16` trên H200/A100, `fp16` trên V100 |
| `per_device_train_batch_size`, `gradient_accumulation_steps` | theo GPU | được auto-cấu hình, không phải khác biệt giữa `v4` và `v5` |

Reward thực sự dùng trong các run v4/v5 là dạng gated, không phải tổng tuyến tính:

```text
R_acc   = 0.5 × ROUGE-1_F1 + 0.5 × ROUGE-L_F1
R_total = R_acc × (1 + 0.3 × R_len + 0.2 × R_sent)
```

`reward_weight_accuracy=0.5` còn trong config để tương thích API nhưng không nhân trực tiếp vào `R_total`.

Các tham số còn lại của GRPO được giữ nguyên trong cùng một nhóm chạy, nên khi so sánh `v4` với `v5` chỉ nên diễn giải là tác động của cả cụm cấu hình.

---

### 2.3 Metric đánh giá

Run `20260622_103706` dùng bốn metric chính:

| Metric | Cách hiểu đúng trong code đánh giá |
|---|---|
| `ROUGE-2` | overlap bigram giữa sinh và tham chiếu |
| `length_error_pct` | sai lệch độ dài tương đối trung bình: `|gen_len - target_len| / target_len * 100` |
| `length_distance` | khoảng cách tuyệt đối giữa `gen_len` và `ref_len`, không phải giữa `gen_len` và `target_len` |
| `avg_gen_length` | độ dài trung bình của đầu ra |

Hai lưu ý quan trọng:

- `length_error_pct` không phải “tỷ lệ vi phạm”. Vì là sai lệch tương đối trung bình nên có thể vượt `100%`.
- `BARTScore` trong run này là `NaN`, nên không dùng để kết luận.

---

## 3. Quá trình huấn luyện: điều gì thực sự đã xảy ra

### 3.1 SFT hội tụ sạch và là nền tảng cho giai đoạn sau

| Run | Train loss đầu -> cuối | Eval loss đầu -> cuối | 4 eval cuối | Kết luận log |
|---|---|---|---|---|
| Qwen3-4B-Base SFT | 1.9133 -> 1.6328 | 1.8008 -> 1.6781 | 1.6783 / 1.6782 / 1.6781 / 1.6781 | Giảm đều, cuối run gần như phẳng |
| Qwen3-4B-Instruct SFT | 2.5510 -> 1.5741 | 1.9376 -> 1.6635 | 1.6638 / 1.6636 / 1.6635 / 1.6635 | Giảm đều, cuối run gần như phẳng |

Diễn giải:

- Cả hai run SFT đều cho thấy loss giảm ổn định và gần như chạm plateau ở cuối epoch 2.
- `Qwen3-4B-Instruct` bắt đầu khó hơn ở loss đầu, nhưng hội tụ về mức eval loss cuối tốt hơn `Qwen3-4B-Base`.
- Với những log hiện có, nhận định hợp lý nhất là SFT đã hoàn thành vai trò “ổn định hóa hành vi theo instruction” trước khi sang GRPO.

### 3.2 GRPO: v4 và v5 là hai cấu hình đáng báo cáo

| Run | K | LR | β | Val reward @200 | Val reward @800 | Xu hướng |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| Base fresh v4 | 4 | 5e-7 | 0.15 | 0.284 | 0.369 | 🟢 Tăng rõ |
| Base fresh v5 | 8 | 2e-6 | 0.04 | 0.450 | 0.578 | 🟢 Tăng rõ |
| Base SFT-init v4 | 4 | 5e-7 | 0.15 | 0.591 | 0.604 | 🟡 Tăng nhẹ |
| Base SFT-init v5 | 8 | 2e-6 | 0.04 | 0.615 | 0.640 | 🟡 Tăng nhẹ |
| Instruct fresh v4 | 4 | 5e-7 | 0.15 | 0.391 | 0.452 | 🟢 Tăng rõ |
| Instruct fresh v5 | 8 | 2e-6 | 0.04 | 0.466 | 0.517 | 🟢 Tăng rõ |
| Instruct SFT-init v4 | 4 | 5e-7 | 0.15 | 0.613 | 0.628 | 🟡 Tăng nhẹ |
| Instruct SFT-init v5 | 8 | 2e-6 | 0.04 | 0.647 | 0.667 | 🟡 Tăng nhẹ |

Diễn giải:

- Các nhánh `fresh` bắt đầu với val reward thấp hơn nhiều so với `SFT-init`, rồi mới tăng dần.
- Các nhánh `SFT-init` bắt đầu ở mức reward cao hơn hẳn và phần lớn chỉ fine-tune thêm ở biên nhỏ.
- `v5` luôn khởi đầu và kết thúc ở mức val reward cao hơn `v4` trong cả bốn cặp so sánh tương ứng.

Tuy vậy, cần giữ kỷ luật diễn giải:

- `v4 -> v5` là thay đổi đồng thời của `K`, `learning rate`, và `beta`.
- Vì vậy báo cáo này chỉ kết luận về **tác động của cả cụm cấu hình v5**, không quy công cho từng tham số riêng lẻ.

### 3.3 Timeline ngắn của các vòng GRPO trước v4/v5

- Bản GRPO đầu (`v1`) cho thấy reward hacking và rollout thoái hóa: `R_acc` gần 0, mô hình có thể sinh blob hoặc lặp vô nghĩa nhưng vẫn ăn điểm ở reward độ dài/số câu.
- `v2` thêm các fix chống degeneration và smoke test đã xác nhận reward không còn bị game theo kiểu cũ.
- `v3` thử `repetition_penalty=1.3` và `no_repeat_ngram_size=3`, nhưng rollout lại hỏng nặng vì cơ chế generate phạt cả việc dùng lại từ vựng của prompt, làm `R_acc` về gần 0.
- `v4` và `v5` là hai cấu hình “sạch” sau khi bỏ hướng xử lý sai ở `v3`; đây là hai phiên bản nên được dùng để báo cáo chính thức.

---

## 4. Kết quả tổng hợp trên toàn bộ tập test

| Mô hình | ROUGE-2 | LenErr% ↓ | LenDist ↓ | AvgLen |
|:---|---:|---:|---:|---:|
| `QWEN3_BASE_base` | 0.1056 | 169.77 | 51.8 | 71.9 |
| `QWEN3_BASE_sft` | 0.2632 | 30.52 | 18.4 | 61.1 |
| `QWEN3_BASE_grpo_fresh_v4` | 0.1388 | 76.44 | 32.8 | 58.2 |
| `QWEN3_BASE_grpo_fresh_v5` | 0.1866 | 29.05 | 21.5 | 50.3 |
| `QWEN3_BASE_grpo_sft_v4` | 0.2671 | 20.82 | 17.1 | 57.3 |
| `QWEN3_BASE_grpo_sft_v5` | 0.2638 | 14.94 | 20.5 | 46.7 |
| `QWEN3_INSTRUCT_base` | 0.1506 | 30.22 | 18.9 | 54.9 |
| `QWEN3_INSTRUCT_sft` | 0.2609 | **12.42** 🥇 | **14.6** 🥇 | 51.5 |
| `QWEN3_INSTRUCT_grpo_fresh_v4` | 0.1551 | 26.06 | 18.3 | 52.9 |
| `QWEN3_INSTRUCT_grpo_fresh_v5` | 0.1669 | 22.78 | 17.7 | 50.7 |
| `QWEN3_INSTRUCT_grpo_sft_v4` | 0.2655 | 13.41 | 16.9 | 48.0 |
| **`QWEN3_INSTRUCT_grpo_sft_v5`** | **0.2765** 🥇 | 15.85 | 18.8 | **44.4** |

Nhận xét cấp cao:

- `QWEN3_INSTRUCT_grpo_sft_v5` có `ROUGE-2` cao nhất toàn run.
- `QWEN3_INSTRUCT_sft` có sai lệch độ dài tương đối thấp nhất toàn run.
- `QWEN3_BASE_grpo_sft_v5` có sai lệch độ dài tương đối tốt nhất trong họ `Base`.
- Hai nhánh `fresh` đều thua rõ các nhánh `SFT-init`, dù `v5` giúp chúng tốt hơn `v4`.

---

## 5. Tác động của từng thành phần post-training

### 5.1 Bảng delta theo các phép so sánh có kiểm soát

| So sánh | ΔROUGE-2 | ΔLenErr% | ΔLenDist | ΔAvgLen |
|:---|---:|---:|---:|---:|
| **Base → SFT** | **+0.1576** 🟢 | **-139.25** 🟢 | **-33.41** 🟢 | -10.74 |
| **Instruct → SFT** | **+0.1104** 🟢 | **-17.80** 🟢 | **-4.38** 🟢 | -3.34 |
| Base → fresh GRPO v5 | +0.0810 🟢 | -140.72 🟢 | -30.33 🟢 | -21.53 |
| Instruct → fresh GRPO v5 | +0.0163 🟢 | -7.44 🟢 | -1.29 🟢 | -4.12 |
| Base SFT → Base SFT+GRPO v5 | +0.0007 🟢 | -15.58 🟢 | +2.08 🔴 | -14.41 |
| Instruct SFT → Instruct SFT+GRPO v5 | +0.0156 🟢 | +3.43 🔴 | +4.25 🔴 | -7.09 |
| Base fresh GRPO v5 → Base SFT+GRPO v5 | +0.0773 🟢 | -14.11 🟢 | -1.01 🟢 | -3.61 |
| Instruct fresh GRPO v5 → Instruct SFT+GRPO v5 | **+0.1097** 🟢 | -6.93 🟢 | +1.15 🔴 | -6.31 |
| Base GRPO SFT-init: v4 → v5 | -0.0032 🔴 | -5.89 🟢 | +3.34 🔴 | -10.60 |
| Instruct GRPO SFT-init: v4 → v5 | +0.0111 🟢 | +2.44 🔴 | +1.93 🔴 | -3.60 |
| Pretrained only: Base → Instruct | +0.0450 🟢 | -139.55 🟢 | -32.85 🟢 | -17.00 |
| After SFT: Base → Instruct | -0.0023 🔴 | -18.10 🟢 | -3.81 🟢 | -9.60 |
| After SFT+GRPO v5: Base → Instruct | +0.0127 🟢 | +0.92 🔴 | -1.65 🟢 | -2.29 |

### 5.2 Diễn giải theo thành phần

**SFT là bước cải thiện lớn nhất và ổn định nhất**

- `Base -> SFT` là bước nhảy mạnh nhất ở họ `Base`: `ROUGE-2` tăng `+0.1576`, sai lệch độ dài giảm `-139.25`.
- `Instruct -> SFT` cũng cải thiện rõ, nhưng biên tăng nhỏ hơn vì bản thân pretrained instruct đã mạnh hơn base ngay từ đầu.
- Đây là lý do có thể kết luận SFT là nền tảng bắt buộc trước khi mong GRPO phát huy hiệu quả.

**GRPO từ pretrained (`fresh`) có ích, nhưng không đủ thay SFT**

- Với họ `Base`, `fresh GRPO v5` tốt hơn pretrained rõ rệt, nhưng vẫn kém `SFT` và kém xa `SFT+GRPO`.
- Với họ `Instruct`, `fresh GRPO v5` chỉ cải thiện nhẹ về `ROUGE-2`, cho thấy GRPO một mình không thay thế được quá trình học bám instruction ở SFT.

**GRPO sau SFT mới là nhánh mạnh nhất**

- So với `fresh GRPO v5`, các nhánh `SFT+GRPO v5` đều tăng mạnh `ROUGE-2`: `+0.0773` ở họ `Base` và `+0.1097` ở họ `Instruct`.
- Điều này củng cố kết luận: GRPO hiệu quả hơn nhiều khi nó tinh chỉnh một chính sách đã biết “tóm tắt theo yêu cầu”, thay vì tự học điều đó từ pretrained model.

**Tác động của GRPO lên SFT là khác nhau giữa Base và Instruct**

- Ở họ `Base`, `SFT -> SFT+GRPO v5` gần như giữ nguyên `ROUGE-2` (`+0.0007`) nhưng kéo sai lệch độ dài tốt hơn đáng kể (`-15.58`).
- Ở họ `Instruct`, `SFT -> SFT+GRPO v5` tăng `ROUGE-2` (`+0.0156`) nhưng làm sai lệch độ dài xấu đi (`+3.43`).
- Nghĩa là với họ `Instruct`, GRPO v5 thiên về nâng chất lượng nội dung hơn là siết chặt length control; với họ `Base`, GRPO v5 thiên về nén độ dài nhiều hơn.

**`v5` tốt hơn `v4`, nhưng lợi ích phụ thuộc nhánh và metric**

- Ở các nhánh `fresh`, `v5` tốt hơn `v4` khá rõ ở cả reward log và metric cuối.
- Ở nhánh `Base SFT-init`, `v5` không tăng `ROUGE-2` so với `v4`, nhưng cải thiện đáng kể độ bám độ dài.
- Ở nhánh `Instruct SFT-init`, `v5` tăng `ROUGE-2`, nhưng trả giá bằng length control kém hơn một chút.
- Vì `v5` thay đổi đồng thời `K=8`, `LR=2e-6`, `beta=0.04`, chỉ nên kết luận rằng **cụm cấu hình v5** hiệu quả hơn cho nhiều tình huống, chứ không khẳng định tham số nào là nguyên nhân chính.

---

## 6. Kết quả theo dataset

### 6.1 ROUGE-2

| Mô hình | VietNews 🏆 | WikiLingua 🏆 | ViMs 🏆 | VLSP 🏆 |
|:---|---:|---:|---:|---:|
| `QWEN3_BASE_base` | 0.1213 | 0.0318 | 0.1346 | 0.0949 |
| `QWEN3_BASE_sft` | 0.2661 | 0.2443 | 0.2565 | **0.2814** 🥇 |
| `QWEN3_BASE_grpo_fresh_v4` | 0.1522 | 0.0664 | 0.1740 | 0.1348 |
| `QWEN3_BASE_grpo_fresh_v5` | 0.2073 | 0.0971 | 0.2028 | 0.1814 |
| `QWEN3_BASE_grpo_sft_v4` | 0.2745 | 0.2505 | **0.2597** 🥇 | 0.2527 |
| `QWEN3_BASE_grpo_sft_v5` | 0.2897 | **0.2635** 🥇 | 0.1816 | 0.1746 |
| `QWEN3_INSTRUCT_base` | 0.1648 | 0.0744 | 0.1783 | 0.1551 |
| `QWEN3_INSTRUCT_sft` | 0.2670 | 0.2393 | 0.2471 | 0.2703 |
| `QWEN3_INSTRUCT_grpo_fresh_v4` | 0.1708 | 0.0769 | 0.1788 | 0.1569 |
| `QWEN3_INSTRUCT_grpo_fresh_v5` | 0.1867 | 0.0809 | 0.1843 | 0.1605 |
| `QWEN3_INSTRUCT_grpo_sft_v4` | 0.2803 | 0.2444 | 0.2177 | 0.2497 |
| **`QWEN3_INSTRUCT_grpo_sft_v5`** | **0.2962** 🥇 | 0.2559 | 0.2127 | 0.2439 |

### 6.2 Sai lệch độ dài tương đối (%) — thấp hơn = tốt hơn

| Mô hình | VietNews | WikiLingua | ViMs | VLSP |
|:---|---:|---:|---:|---:|
| `QWEN3_BASE_base` | 205.89 | 174.67 | 42.61 | 47.93 |
| `QWEN3_BASE_sft` | 31.19 | 24.72 | 38.69 | 27.53 |
| `QWEN3_BASE_grpo_fresh_v4` | 81.16 | 102.18 | 37.37 | 41.17 |
| `QWEN3_BASE_grpo_fresh_v5` | 22.36 | 50.94 | 31.89 | 34.33 |
| `QWEN3_BASE_grpo_sft_v4` | 17.10 | 18.66 | 39.62 | 30.43 |
| `QWEN3_BASE_grpo_sft_v5` | 6.43 | **14.69**🥇 | 44.42 | 42.55 |
| `QWEN3_INSTRUCT_base` | 24.50 | 57.47 | **26.07** 🥇 | 27.09 |
| `QWEN3_INSTRUCT_sft` | 6.26 | 18.34 | 31.55 | **24.51** 🥇 |
| `QWEN3_INSTRUCT_grpo_fresh_v4` | 20.15 | 47.81 | 27.63 | 27.64 |
| `QWEN3_INSTRUCT_grpo_fresh_v5` | 17.08 | 38.67 | 28.37 | 28.76 |
| `QWEN3_INSTRUCT_grpo_sft_v4` | **6.00** 🥇 | 20.18 | 36.24 | 28.69 |
| `QWEN3_INSTRUCT_grpo_sft_v5` | 9.24 | 18.11 | 39.14 | 32.85 |

### 6.3 Length distance — thấp hơn = tốt hơn

| Mô hình | VietNews 🥇 | WikiLingua 🥇 | ViMs 🥇 | VLSP 🥇 |
|:---|---:|---:|---:|---:|
| `QWEN3_BASE_base` | 33.0 | 61.6 | 92.0 | 120.3 |
| `QWEN3_BASE_sft` | 5.7 | 14.2 | 66.4 | **61.6** 🥇 |
| `QWEN3_BASE_grpo_fresh_v4` | 13.1 | 40.3 | 82.7 | 102.3 |
| `QWEN3_BASE_grpo_fresh_v5` | 3.8 | 22.0 | 71.8 | 88.0 |
| `QWEN3_BASE_grpo_sft_v4` | 3.2 | 10.9 | 69.4 | 68.1 |
| `QWEN3_BASE_grpo_sft_v5` | 1.2 | **8.4** 🥇 | 89.2 | 100.5 |
| `QWEN3_INSTRUCT_base` | 4.0 | 23.4 | **59.8** 🥇 | 70.2 |
| `QWEN3_INSTRUCT_sft` | 1.1 | 9.9 | 64.5 | 62.2 |
| `QWEN3_INSTRUCT_grpo_fresh_v4` | 3.4 | 19.9 | 62.3 | 70.9 |
| `QWEN3_INSTRUCT_grpo_fresh_v5` | 2.9 | 16.1 | 63.1 | 72.9 |
| `QWEN3_INSTRUCT_grpo_sft_v4` | **1.0** 🥇 | 11.0 | 76.1 | 73.0 |
| `QWEN3_INSTRUCT_grpo_sft_v5` | 1.5 | 10.2 | 84.5 | 82.7 |

### 6.4 Những gì các bảng trên nói về từng nhóm dữ liệu

**Single-document (VietNews, WikiLingua)**

- Đây là nơi pipeline hiện tại mạnh nhất.
- `QWEN3_INSTRUCT_grpo_sft_v5` đứng đầu trên VietNews về `ROUGE-2` (`0.2962`).
- `QWEN3_BASE_grpo_sft_v5` đứng đầu trên WikiLingua về cả `ROUGE-2` (`0.2635`) lẫn hai metric độ dài.
- Với single-document, các nhánh `SFT-init` rõ ràng tốt hơn `fresh`, và độ bám độ dài đã xuống vùng rất thấp (`~1-10` ở length distance cho các model tốt nhất).

**Multi-document (ViMs, VLSP)**

- Kết quả không còn có một “mô hình thắng tuyệt đối” trên mọi metric.
- ViMs có `ROUGE-2` tốt nhất ở `QWEN3_BASE_grpo_sft_v4` (`0.2597`), nhưng metric độ dài tốt nhất lại nằm ở `QWEN3_INSTRUCT_base` chứ không phải mô hình hậu huấn luyện sâu hơn.
- VLSP có `ROUGE-2` tốt nhất ở `QWEN3_BASE_sft` (`0.2814`) và length distance tốt nhất cũng ở model này (`61.6`), trong khi `QWEN3_INSTRUCT_sft` lại tốt nhất về sai lệch độ dài tương đối (`24.51`).
- Nói ngắn gọn: single-document cho thấy lợi ích post-training rất rõ; multi-document vẫn là phần khó và chưa cho thấy một chiến lược hậu huấn luyện thắng nhất quán.

---

## 7. So sánh với các baseline tham khảo

> **Lưu ý trước khi so sánh:**
> - Dữ liệu tham khảo từ `VDT_Textsum/ketqua.md`. Bảng tham khảo dùng `ROUGE-2` theo đơn vị phần trăm (%); các số từ run hiện tại đã được nhân `100` để cùng thang đo.
> - Chưa có bảo đảm hai bên dùng cùng split, cùng prompt template và cùng decoding setup.
> - Vì run hiện tại không có `BARTScore` hợp lệ, phần so sánh chỉ dùng `ROUGE-2` và `length distance`.
> - Do thiếu provenance và protocol chung, phần này chỉ cung cấp bối cảnh số liệu. Không dùng các bảng dưới đây để tuyên bố model project vượt hoặc kém một baseline ngoài project.

### 7.1 ROUGE-2 (%) — So sánh toàn diện

Bảng dưới đây so sánh tất cả biến thể Qwen3-4B trong run này với các mô hình tham khảo. Các ô được tô màu: **🟢 xanh** = top 3, **🟡 vàng** = top 5, còn lại = không highlight. 🥇 = tốt nhất dataset đó.

| Nhóm | Mô hình | WikiLingua | VietNews | ViMs | VLSP |
|:---|:---|---:|---:|---:|---:|
| **🏆 Tham khảo mạnh** | VietAI | **33.12** 🥇 | **34.24** 🥇 | — | — |
| | GPT-4o | 20.65 | 21.61 | **44.26** 🥇 | 43.37 |
| | Qwen3-14B | 20.24 | 18.51 | **44.38** 🥇 | **44.27** 🥇 |
| | Llama3.3-70B-Instruct | 19.40 | 22.17 | 37.54 | 39.02 |
| | Phi4-14B | 14.28 | 13.17 | 41.85 | 42.31 |
| | Sailor-20B-chat | 18.74 | 17.70 | 39.47 | 40.96 |
| | VinBigdata (7B) | 21.02 | 20.59 | 37.98 | 40.23 |
| | gpt-3.5-turbo | 21.09 | 28.13 | 19.28 | 35.79 |
| **Tham khảo cùng cỡ** | Qwen3-4B (base) | 21.25 | 19.05 | 19.57 | 42.09 |
| | Qwen3-0.6B | 20.07 | 18.27 | 17.49 | 40.51 |
| **Qwen3-Base<br>Run này** | Base (pretrained) | 3.18 | 12.13 | 13.46 | 9.49 |
| | + SFT | 24.43 🟡 | 26.61 🟡 | 25.65 🟡 | **28.14** 🟢 |
| | + GRPO SFT-init v4 | 25.05 🟡 | 27.45 🟡 | **25.97** 🟡 | 25.27 |
| | + GRPO SFT-init v5 | **26.35** 🟢 | 28.97 🟢 | 18.16 | 17.46 |
| **Qwen3-Instruct<br>Run này** | Base (pretrained) | 7.44 | 16.48 | 17.83 | 15.51 |
| | + SFT | 23.93 🟡 | 26.70 🟡 | 24.71 🟡 | 27.03 🟡 |
| | + GRPO SFT-init v4 | 24.44 🟡 | 28.03 🟢 | 21.77 | 24.97 |
| | + GRPO SFT-init v5 | 25.59 🟢 | **29.62** 🟢 | 21.27 | 24.39 |

> **Cách đọc:** 🟢 = top 3 toàn bảng, 🟡 = top 5. Giá trị **in đậm** = tốt nhất trong nhóm run này. 🥇 = tốt nhất dataset.

**Nhận xét ROUGE-2:**

- **Single-document (WikiLingua, VietNews):** Các biến thể Qwen3-4B sau post-training đạt 23–30% trong protocol nội bộ. Các số tham khảo nằm ở khoảng 18–34%, nhưng không thể xếp hạng trực tiếp khi split, prompt và metric implementation chưa được đồng nhất.

- **Multi-document (ViMs, VLSP):** Các số nội bộ thấp hơn phần lớn số trong bảng tham khảo. Quan trọng hơn, ViMs/VLSP của run này bị overlap với train/validation, nên không dùng để đánh giá generalization hoặc so sánh baseline.

### 7.2 Length Distance — So sánh khả năng tuân thủ độ dài

(Thấp hơn = tốt hơn. 🥇 = tốt nhất dataset. Dấu `—` = không có dữ liệu.)

| Nhóm | Mô hình | WikiLingua | VietNews | ViMs | VLSP |
|:---|:---|---:|---:|---:|---:|
| **🏆 Tham khảo** | GPT-4o | 11 | 8 | 187 | 58 |
| | gpt-3.5-turbo | 17 | 18 | **47** 🥇 | **30** 🥇 |
| | Qwen3-14B | 45 | 50 | 131 | 34 |
| | Llama3.3-70B-Instruct | 13 | 14 | 186 | 73 |
| | Sailor2-20B-Chat | 43 | 38 | 268 | 41 |
| | Phi4-14B | 214 | 205 | 166 | 134 |
| | VinBigdata (7B) | 89 | 115 | 98 | 82 |
| **Qwen3-Base<br>Run này** | Base (pretrained) | 61.6 | 33.0 | 92.0 | 120.3 |
| | + SFT | 14.2 🟢 | 5.7 🟢 | 66.4 🟡 | **61.6** 🟡 |
| | + GRPO SFT-init v4 | 10.9 🟢 | 3.2 🟢 | 69.4 | 68.1 |
| | + GRPO SFT-init v5 | **8.4** 🟢 | 1.2 🟢 | 89.2 | 100.5 |
| **Qwen3-Instruct<br>Run này** | Base (pretrained) | 23.4 🟡 | 4.0 🟢 | **59.8** 🟡 | 70.2 |
| | + SFT | 9.9 🟢 | 1.1 🟢 | 64.5 🟡 | 62.2 🟡 |
| | + GRPO SFT-init v4 | 11.0 🟢 | **1.0** 🟢 | 76.1 | 73.0 |
| | + GRPO SFT-init v5 | 10.2 🟢 | 1.5 🟢 | 84.5 | 82.7 |

> **Cách đọc màu:** 🟢 = trong top 3 thấp nhất toàn bảng, 🟡 = top 5.

**Nhận xét Length Distance:**

- **Single-document:** Trên VietNews và WikiLingua, các biến thể SFT/GRPO của Qwen3-4B đạt LenDist thấp (1.0–14.2) trong protocol nội bộ. Việc so sánh trực tiếp với GPT-4o/Llama chỉ hợp lệ sau khi dùng cùng prompt, split và cách đếm độ dài.

- **Multi-document — vẫn là thách thức:** Trên ViMs và VLSP, tất cả mô hình (cả tham khảo lẫn run này) đều có LenDist cao (30–268). Các mô hình run này (60–100) nằm ở vùng trung bình, chưa tốt bằng gpt-3.5-turbo (30–47). Multi-document summarization với kiểm soát độ dài vẫn là bài toán mở.

### 7.3 Tổng kết so sánh

| Tiêu chí | Mô hình tốt nhất run này | Giá trị run này | Tham khảo tốt nhất | Chênh lệch |
|:---|---:|---:|---:|---:|
| **ROUGE-2** WikiLingua | `QWEN3_BASE_grpo_sft_v5` | 26.35% | VietAI: 33.12% | **-6.77pp** 🔴 |
| **ROUGE-2** VietNews | `QWEN3_INSTRUCT_grpo_sft_v5` | 29.62% | VietAI: 34.24% | **-4.62pp** 🔴 |
| **ROUGE-2** ViMs | `QWEN3_BASE_grpo_sft_v4` | 25.97% | Qwen3-14B: 44.38% | **-18.41pp** 🔴 |
| **ROUGE-2** VLSP | `QWEN3_BASE_sft` | 28.14% | Qwen3-14B: 44.27% | **-16.13pp** 🔴 |
| **LenDist** WikiLingua | `QWEN3_BASE_grpo_sft_v5` | **8.4** 🥇 | GPT-4o: 11 | **-2.6** 🟢 |
| **LenDist** VietNews | `QWEN3_INSTRUCT_grpo_sft_v4` | **1.0** 🥇 | GPT-4o: 8 | **-7.0** 🟢 |
| **LenDist** ViMs | `QWEN3_INSTRUCT_base` | 59.8 | gpt-3.5-turbo: 47 | **+12.8** 🔴 |
| **LenDist** VLSP | `QWEN3_BASE_sft` | 61.6 | gpt-3.5-turbo: 30 | **+31.6** 🔴 |

> 🟢 = run này tốt hơn tham khảo tốt nhất 🔴 = run này kém hơn tham khảo tốt nhất  
> pp = percentage points (chênh lệch phần trăm tuyệt đối)

**Ba kết luận chính từ so sánh:**

1. **Single-document là kết quả mạnh nhất của project.** Pipeline SFT+GRPO cải thiện rõ so với pretrained/SFT baselines nội bộ. Chưa đủ điều kiện để tuyên bố vượt Qwen3-14B, Llama3.3-70B hoặc GPT-4o.

2. **Multi-document: còn cách biệt lớn.** Trên ViMs và VLSP, ROUGE-2 của 4B thua các mô hình 14B+ từ 16–18pp. LenDist cũng chưa cạnh tranh được với gpt-3.5-turbo. Đây là hướng cần cải thiện trong các phiên bản sau.

3. **Khoảng cách với VietAI:** VietAI (mô hình chuyên tóm tắt tiếng Việt) vẫn dẫn đầu ROUGE-2 trên single-document (33–34%). Khoảng cách 4–7pp cho thấy tiềm năng cải thiện nếu tiếp tục tinh chỉnh pipeline.

---

## 8. Mô hình đã đạt yêu cầu bài toán đến mức nào

### 8.1 Về chất lượng tóm tắt

- Có thể khẳng định pipeline hiện tại đã tạo ra bước tiến rõ rệt so với pretrained model.
- `ROUGE-2` toàn tập tăng từ `0.1056 -> 0.2632/0.2671/0.2638` ở họ `Base` và từ `0.1506 -> 0.2609/0.2655/0.2765` ở họ `Instruct`.
- Trên single-document, mức chất lượng đạt được là thuyết phục.
- Trên multi-document, chất lượng chưa ổn định và chưa tiệm cận các mốc tham khảo mạnh hơn.

### 8.2 Về kiểm soát độ dài

- Có bằng chứng rõ rằng pipeline đã học được length control.
- Từ pretrained sang SFT/GRPO, cả `length_error_pct` lẫn `length_distance` đều giảm mạnh ở phần lớn nhánh.
- Tuy nhiên, khả năng kiểm soát độ dài không đồng đều giữa các dataset:
  - rất tốt ở `VietNews` và `WikiLingua`,
  - còn yếu hơn đáng kể ở `ViMs` và `VLSP`.

### 8.3 Về kiểm soát số câu

- Có reward và cấu hình huấn luyện cho `sentence control`.
- Nhưng ở run đánh giá cuối hiện tại, chưa có metric tổng kết riêng để chứng minh mức độ tuân thủ số câu.
- Vì vậy chưa nên kết luận rằng yêu cầu `sentence control` đã được đáp ứng đầy đủ.

### 8.4 Kết luận ngắn cho yêu cầu bài toán

Đánh giá trung thực nhất là:

- **Đã đạt tiến bộ rõ rệt** ở hai mặt `quality` và `length control`.
- **Đã có hệ thống huấn luyện hỗ trợ `sentence control`**, nhưng **chưa có bằng chứng đánh giá cuối đủ mạnh** để coi đây là mục tiêu đã hoàn thành.
- **Single-document** là vùng đã làm khá tốt.
- **Multi-document** vẫn là phần chưa đạt kỳ vọng nếu so với các mốc tham khảo mạnh.

---

## 9. Kết luận chính

1. `SFT` là thành phần tạo ra bước nhảy lớn nhất và là nền bắt buộc cho `GRPO`.
2. `GRPO` khởi tạo từ pretrained (`fresh`) cải thiện được, nhưng không hiệu quả bằng `GRPO` khởi tạo từ `SFT`.
3. `v5` nhìn chung tốt hơn `v4`, nhưng lợi ích không đồng nhất trên mọi nhánh và không thể tách nhân quả cho từng hyperparameter vì đây là thay đổi dạng bundle.
4. `QWEN3_INSTRUCT_grpo_sft_v5` là model có chất lượng tổng thể cao nhất trong run này nếu ưu tiên `ROUGE-2`.
5. Nếu ưu tiên bám độ dài chặt hơn, một số cấu hình khác vẫn đáng giữ lại để so sánh, đặc biệt `QWEN3_INSTRUCT_sft`, `QWEN3_INSTRUCT_grpo_sft_v4`, và `QWEN3_BASE_grpo_sft_v5`.
6. Pipeline hiện tại đã có bằng chứng tốt cho single-document summarization có ràng buộc độ dài, nhưng chưa đủ mạnh để tuyên bố đã giải tốt phần multi-document và sentence adherence.
