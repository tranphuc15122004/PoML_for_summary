# Mô tả chi tiết các bộ dữ liệu — PoML for Summary

> Tài liệu tham khảo nhanh cho mọi giai đoạn triển khai.
> Không cần tra cứu code hay data gốc mỗi lần chạy experiment.

---

## Tổng quan

| Dataset | Loại | Ngôn ngữ | Ước tính mẫu | Nguồn |
|---------|------|----------|-------------|-------|
| VietNews | Single-doc, abstractive | Vietnamese | ~150K | Báo điện tử Việt Nam |
| WikiLingua | Single-doc, abstractive | Vietnamese | ~19.5K | WikiHow (Vietnamese subset) |
| ViMs | Multi-doc, abstractive | Vietnamese | 300 clusters | Tổng hợp từ nhiều nguồn |
| VLSP | Multi-doc, extractive | Vietnamese | ~600 | VLSP 2022 AbMuSu shared task |

**Tổng hợp:** ~170K mẫu thô → sau augmentation SFT: ~510K (×3 variants), GRPO: ~8K prompts.

---

## 1. VietNews

### 1.1. Mô tả

Bộ dữ liệu tóm tắt tin tức tiếng Việt single-document. Mỗi mẫu gồm một bài báo và title được dùng làm tóm tắt gold. Đây là nguồn dữ liệu lớn nhất và là backbone chính cho SFT.

### 1.2. Cấu trúc thư mục

```
VDT_Textsum/vietnews-master/vietnews-master/data/
├── train_tokenized/     # ~130K file .seg
├── val_tokenized/       # ~10K file .seg
└── test_tokenized/      # ~10K file .seg
```

### 1.3. Định dạng file `.seg`

Mỗi file `.txt.seg` có cấu trúc:

```
Line 1:  Tiêu đề (title)          → dùng làm target/summary
Line 2:  (trống)
Line 3:  Sapo/Lead                → source (phần 1)
Line 4:  (trống)
Line 5+: Body                     → source (phần 2)
Dòng cuối: Caption/credit         → BỎ qua, không dùng
```

### 1.4. Xử lý data

- **Source:** Sapo + Body (nối bằng newline), bỏ caption dòng cuối
- **Target:** Title (dòng đầu tiên)
- **Truncation:** source ≤ 8000 chars, summary ≤ 1500 chars
- **Preprocessing:** Thay underscore `_` bằng space (do word segmentation markers từ underthesea)

### 1.5. Thống kê (ước tính)

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Số mẫu | ~130K | ~10K | ~10K |
| Độ dài source (từ) | trung bình ~300-500 | tương tự | tương tự |
| Độ dài summary (từ) | trung bình ~20-30 (title) | tương tự | tương tự |
| Loại tóm tắt | Abstractive (title) | | |

### 1.6. Sử dụng trong pipeline

| Giai đoạn | Phần sử dụng | Cách dùng |
|-----------|--------------|-----------|
| SFT Round A | train (raw) | 1:1 mapping, không augment |
| SFT Round B | train (augmented ×3) | + length/style instructions |
| SFT Val | val (first 2K) | Evaluate SFT |
| GRPO Train | val (remaining ~8K) | Prompt cho rollout |
| GRPO Val | val subset | Validate GRPO |
| Test | test (first 2K) | Đánh giá cuối cùng |

### 1.7. Lưu ý quan trọng

- Target là **title**, không phải summary → rất ngắn (1-2 câu)
- Một số file có encoding lỗi → cần try/except khi đọc
- Caption dòng cuối đôi khi chứa text → phải skip

---

## 2. WikiLingua

### 2.1. Mô tả

Bộ dữ liệu tóm tắt từ WikiHow — hướng dẫn "how-to" đa dạng domain (nấu ăn, sửa chữa, sức khỏe, v.v.). Mỗi mẫu gồm một đoạn hướng dẫn và tóm tắt tương ứng. cung cấp sự đa dạng về domain so với VietNews (tin tức).

### 2.2. Cấu trúc thư mục

```
VDT_Textsum/wikilingua/wikilingua/
├── train.json
├── val.json
└── test.json
```

### 2.3. Định dạng JSON

Mỗi dòng là một JSON object (JSONL format):

```json
{
  "src": ["sentence1", "sentence2", ...],
  "tgt": ["summary_sentence1", ...]
}
```

- `src`: List các câu trong bài hướng dẫn gốc
- `tgt`: List các câu trong bản tóm tắt

### 2.4. Xử lý data

- **Source:** Nối tất cả elements trong `src` bằng space
- **Target:** Nối tất cả elements trong `tgt` bằng space
- **Truncation:** source ≤ 8000 chars, summary ≤ 1500 chars
- **Preprocessing:** Thay underscore `_` bằng space

### 2.5. Thống kê (ước tính)

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Số mẫu | ~15K | ~2K | ~2.5K |
| Độ dài source (từ) | trung bình ~150-300 | tương tự | tương tự |
| Độ dài summary (từ) | trung bình ~30-50 | tương tự | tương tự |
| Domain | Đa dạng (how-to) | | |

### 2.6. Sử dụng trong pipeline

| Giai đoạn | Phần sử dụng | Cách dùng |
|-----------|--------------|-----------|
| SFT Round A | train (raw) | 1:1 mapping, không augment |
| SFT Round B | train (augmented ×3) | + length/style instructions |
| SFT Val | val (first 500) | Evaluate SFT |
| GRPO Train | val (remaining) | Prompt cho rollout |
| Test | test (first 500) | Đánh giá cuối cùng |

### 2.7. Lưu ý quan trọng

- Dữ liệu dạng list sentences, cần join đúng cách
- WikiHow content thường dài hơn news titles → summary cũng dài hơn
- Ít samples hơn VietNews nhưng đa dạng domain hơn

---

## 3. ViMs

### 3.1. Mô tả

Bộ dữ liệu tóm tắt đa tài liệu (multi-document) tiếng Việt. Mỗi "cluster" gồm nhiều bài báo liên quan về cùng một sự kiện/sự việc, kèm gold summary từ annotators. Đây là dữ liệu challenging nhất vì cần tổng hợp thông tin từ nhiều nguồn.

### 3.2. Cấu trúc thư mục

```
VDT_Textsum/ViMs-Dataset-master/ViMs-Dataset-master/ViMs/ViMs/
├── original/                    # 300 clusters
│   ├── Cluster_001/
│   │   └── original/
│   │       ├── 0.txt           # Document 1
│   │       ├── 1.txt           # Document 2
│   │       └── ...
│   ├── Cluster_002/
│   │   └── original/
│   └── ...
└── summary/                     # Gold summaries
    ├── Cluster_001/
    │   ├── 0.gold.txt          # Annotator 0
    │   └── 1.gold.txt          # Annotator 1
    ├── Cluster_002/
    └── ...
```

### 3.3. Định dạng file document

Mỗi file `.txt` trong `original/` có metadata header + content:

```
Title: [tiêu đề bài báo]
Source: [nguồn]
Link: [URL]
Published Date: [ngày]
Author: [tác giả]
Tags: [thẻ]
Summary: [tóm tắt gốc]
Content:
[Nội dung bài báo đầy đủ]
```

Metadata được parse tự động bởi regex `^([A-Za-z][A-Za-z _]*?):\s*(.*)$`.

### 3.4. Định dạng gold summary

- File `0.gold.txt`: Gold summary từ annotator thứ nhất
- File `1.gold.txt`: Gold summary từ annotator thứ hai
- Mỗi file chứa plain text summary

### 3.5. Xử lý data

- **Source:** Tất cả documents trong cluster, mỗi document Format:
  ```
  [Tài liệu 1] Title
  Content...

  ---

  [Tài liệu 2] Title
  Content...
  ```
- **Target:** Gold summary (chọn annotator_idx=0 hoặc 1)
- **Truncation:** source ≤ 8000 chars, summary ≤ 1500 chars
- **Instruction template:** Dùng `multi_doc_instruction` (khác single-doc)

### 3.6. Thống kê (ước tính)

| Metric | Giá trị |
|--------|---------|
| Số clusters | 300 |
| Documents/cluster | trung bình 5-10 |
| Tổng documents | ~2,000-3,000 |
| Độ dài source (từ) | trung bình ~1000-2000 (tổng hợp nhiều docs) |
| Độ dài summary (từ) | trung bình ~50-100 |
| Annotators | 2 (0 và 1) |

### 3.7. Sử dụng trong pipeline

| Giai đoạn | Phần sử dụng | Cách dùng |
|-----------|--------------|-----------|
| GRPO Train | 80% clusters (~240) | Prompt cho rollout |
| GRPO Val | 20% clusters (~60) | Validate GRPO |
| Test | 300 clusters (full) | Đánh giá cuối cùng |

### 3.8. Lưu ý quan trọng

- **KHÔNG CÓ train/val/test split** — dùng entire dataset
- Source rất dài do gộp nhiều documents → hay bị truncate
- Dùng `annotator_idx=0` làm default (có thể test cả 2 annotators)
- Multi-doc cần `max_seq_length=4096` khi train

---

## 4. VLSP

### 4.1. Mô tả

Bộ dữ liệu tóm tắt đa tài liệu extractive từ VLSP 2022 AbMuSu shared task. Khác với ViMs (abstractive), VLSP yêu cầu **chọn các câu quan trọng** từ source documents (extractive summarization). Labels là chỉ số index của các câu được chọn.

### 4.2. Cấu trúc thư mục

```
VDT_Textsum/vlsp/vlsp/
├── train.label.jsonl
├── val.label.jsonl
├── test.label.jsonl
└── vlsp_2022_abmusu.label.jsonl
```

### 4.3. Định dạng JSONL

```json
{
  "id": 1,
  "text": [
    [title_doc1, sent1, sent2, ...],
    [title_doc2, sent1, sent2, ...],
    ...
  ],
  "label": [3, 7, 12, ...]
}
```

- `text`: List of documents, mỗi document là list [title, sentence1, sentence2, ...]
- `label`: Flat list indices của các câu được chọn làm summary
- `test.label.jsonl`: Labels placeholder = [0] (không có gold)

### 4.4. Xử lý data

- **Source:** Mỗi document format:
  ```
  [Tài liệu 1] Title
  sentence1 sentence2 ...
  ```
  Các documents nối bằng `\n\n---\n\n`

- **Target:** Nối các câu được chọn bởi `label` indices:
  - Flat index mapping: title_doc1 = index 0, sent1_doc1 = index 1, ...
  - `summary_sentences = [flat_sentences[i] for i in labels]`
  - Nối bằng space

- **Instruction template:** Dùng `multi_doc_instruction`

### 4.5. Thống kê (ước tính)

| Split | Số mẫu | Ghi chú |
|-------|--------|---------|
| Train | ~285 | Có labels đầy đủ |
| Val | ~15 | Có labels đầy đủ |
| Test | ~200 | Labels placeholder = [0] |
| AbMuSu | ~100 | Competition test set |

| Metric | Giá trị |
|--------|---------|
| Documents/sample | trung bình 5-10 |
| Độ dài source (từ) | trung bình ~500-1000 |
| Độ dài summary (từ) | trung bình ~30-60 (extractive) |

### 4.6. Sử dụng trong pipeline

| Giai đoạn | Phần sử dụng | Cách dùng |
|-----------|--------------|-----------|
| GRPO Train | train (285 samples) | Prompt cho rollout |
| GRPO Val | val (15 samples) | Validate GRPO |
| Test | test + abmusu | Đánh giá cuối cùng |

### 4.7. Lưu ý quan trọng

- **Extractive** → target là các câu gốc, không phải paraphrase
- Test set không có labels thực → chỉ dùng để generate, không evaluate trực tiếp
- Rất ít samples (~600 total) → chủ yếu dùng cho GRPO curriculum stage
- Labels là flat indices → cần map đúng khi reconstruct summary

---

## 5. Cách các datasets được gộp cho từng giai đoạn

### 5.1. SFT Train (no augmentation)

```
sft_train_no_aug.jsonl = VietNews/train (~130K) + WikiLingua/train (~15K)
                        ≈ 145K samples
Format: {"messages": [system, user, assistant], "meta": {...}}
```

### 5.2. SFT Train (augmented)

```
sft_train.jsonl = (VietNews/train + WikiLingua/train) × 3 variants
                ≈ 145K × 3 = 435K samples
Format: {"messages": [system, user_with_length_style, assistant], "meta": {length_requirement, style}}
```

Mỗi variant có:
- Length requirement ngẫu nhiên: "khoảng X từ", "trong khoảng lo-hi từ", "không quá X từ"
- Style ngẫu nhiên: "báo chí", "trang trọng", "học thuật", "ngắn gọn súc tích", "dạng gạch đầu dòng"

### 5.3. SFT Val

```
sft_val.jsonl = VietNews/val (first 2K) × 3 + WikiLingua/val (first 500) × 3
              ≈ 7,500 samples
Format: tương tự sft_train.jsonl
```

### 5.4. GRPO Train

```
grpo_train.jsonl = VietNews/val (remaining ~8K)
                 + WikiLingua/val (remaining ~1.5K)
                 + VLSP/train (~285)
                 + ViMs 80% (~240 clusters)
                 ≈ 10K samples
Format: {"prompt": [system, user_with_length_style], "reference": gold_summary, "meta": {...}}
```

### 5.5. GRPO Val

```
grpo_val.jsonl = VLSP/val (~15)
               + ViMs 20% (~60 clusters)
               ≈ 75 samples
Format: tương tự grpo_train.jsonl
```

### 5.6. Test

```
test.jsonl = VietNews/test (first 2K)
           + WikiLingua/test (first 500)
           + VLSP/test (~200)
           + VLSP/abmusu (~100)
           ≈ 2,800 samples
Format: {"prompt": [system, user_fixed_style_length], "reference": gold_summary, "meta": {...}}
```

---

## 6. Hệ thống đếm từ (Word Counting)

**QUAN TRỌNG:** Toàn bộ pipeline dùng **syllable-level counting** (đếm tiếng), KHÔNG phải word-level.

```python
word_count = len(text.split())  # Đếm tokens phân tách bởi space
```

Ví dụ:
- "nghiên cứu viên" = 3 tokens (nghiên/cứu/viên), nhưng 1 "word" theo nghĩa thông thường
- "Trường Đại học Bách Khoa" = 5 tokens

**Lý do:** Thống nhất với các baseline tiếng Việt (BARTpho, ViT5, VLSP) — tất cả dùng syllable-level ROUGE.

**Hướng dẫn prompt:** Trong instructions, "từ" luôn hiểu là "tiếng/âm tiết" (syllable).

---

## 7. Style Templates

### 7.1. SFT Styles (5 loại)

| Style | Mô tả | Ví dụ |
|-------|-------|-------|
| `báo chí` | Văn phong báo chí tiêu chuẩn | "Theo thông tin từ..., sự việc xảy ra vào..." |
| `trang trọng` | Formal, nghiêm túc | "Sự kiện trên được ghi nhận và phân tích bởi..." |
| `học thuật` | Academic, có phân tích | "Nghiên cứu chỉ ra rằng..., điều này cho thấy..." |
| `ngắn gọn súc tích` | Ngắn gọn, đi thẳng vào vấn đề | "Sự việc: ..., Kết quả: ..." |
| `dạng gạch đầu dòng` | Bullet points | "- Điểm chính 1\n- Điểm chính 2" |

### 7.2. GRPO Styles (9 loại)

Bao gồm 5 styles của SFT + 4 styles mở rộng:

| Style | Mô tả |
|-------|-------|
| `hài hước` | Humor, dí dỏm |
| `thân mật` | Informal, thân thiện |
| `dành cho trẻ em` | Đơn giản, dễ hiểu |
| `mang tính phản biện` | Critical analysis |

### 7.3. Length Templates (3 loại)

| Template | Format | Tolerance |
|----------|--------|-----------|
| `khoảng X từ` | "khoảng 50 từ" | ±20% (40-60 từ) |
| `trong khoảng lo-hi từ` | "trong khoảng 40-60 từ" | Exact range |
| `không quá X từ` | "không quá 60 từ" | Max limit |

---

## 8. System Prompts

### 8.1. SFT System Prompt

```
Bạn là một trợ lý AI chuyên tóm tắt văn bản tiếng Việt. Hãy tạo ra bản tóm tắt ngắn gọn, chính xác, tuân thủ đúng yêu cầu về độ dài và phong cách.
```

### 8.2. GRPO System Prompt

```
Bạn là một trợ lý AI chuyên tóm tắt văn bản tiếng Việt. Hãy tạo ra bản tóm tắt chính xác, tuân thủ yêu cầu.
```

### 8.3. No-augmentation System Prompt

```
Bạn là một trợ lý AI chuyên tóm tắt văn bản tiếng Việt. Hãy tóm tắt văn bản được cung cấp một cách ngắn gọn và chính xác.
```

---

## 9. Instruction Templates

### 9.1. Single-doc (VietNews, WikiLingua)

```
Hãy tóm tắt văn bản sau đây một cách ngắn gọn:

{source}
```

### 9.2. Multi-doc (ViMs, VLSP)

```
Hãy tóm tắt các văn bản sau đây thành một bản tóm tắt duy nhất, bao quát được các ý chính từ tất cả các văn bản:

{source}
```

### 9.3. Augmented Instruction (SFT)

```
Yêu cầu:
- Độ dài: {length_requirement}
- Phong cách: {style}

Văn bản:
{source}
```

### 9.4. No-augmentation Instruction

```
Tóm tắt văn bản sau:

{source}
```

---

## 10. Data Splitting Strategy

### 10.1. Tại sao split như vậy?

| Nguyên tắc | Giải thích |
|------------|-----------|
| **SFT = single-doc** | VietNews + WikiLingua — dễ học, nhiều samples |
| **GRPO = mixed** | còn lại của val + multi-doc — challenging, cần RL |
| **Test = held-out** | Mỗi dataset có split test riêng, không trùng train |
| **ViMs = 80/20** | Không có split gốc → tự chia cho GRPO train/val |

### 10.2. Flow data

```
Raw Data (VDT_Textsum/)
    │
    ├─[augmenter.py]──► sft_train.jsonl (×3 variants)
    │                   sft_val.jsonl
    │
    ├─[augmenter.py]──► grpo_train.jsonl (prompt-only, no assistant)
    │                   grpo_val.jsonl
    │
    └─[augmenter.py]──► test.jsonl (fixed style+length, no randomness)
```

### 10.3. Quick reference

| Muốn chạy gì? | Cần data gì? | Command |
|---------------|-------------|---------|
| SFT cơ bản | `sft_train_no_aug.jsonl` | `python prepare_no_aug.py` |
| SFT augmented | `sft_train.jsonl` | `python augmenter.py` |
| GRPO | `grpo_train.jsonl`, `grpo_val.jsonl` | `python augmenter.py` |
| Eval | `test.jsonl` | `python augmenter.py` |
| Tất cả | 5 files trong `data/` | `python augmenter.py` |

---

## 11. Troubleshooting

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|------------|-----------|
| VietNews file không đọc được | Encoding error | Try/except, skip file đó |
| WikiLingua `src` không phải list | Dữ liệu corrupt | Check `isinstance(src, list)` |
| ViMs找不到gold summary | Thiếu file `0.gold.txt` | Fallback: dùng file `.gold.txt` đầu tiên tìm thấy |
| VLSP test target rỗng | Labels placeholder = [0] | Expected behavior — test set không có gold |
| Word count lệch | Nhầm syllable vs word | Nhớ: `split()` đếm syllables, không phải words |
| GRPO prompt thiếu samples | Val splits quá nhỏ | Kiểm tra split sizes trước khi train |

---

## 12. Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│  DATASET QUICK REFERENCE                                      │
├──────────────┬──────────┬──────────┬───────────────────────┤
│ Dataset      │ Type     │ Samples  │ Used for               │
├──────────────┼──────────┼──────────┼───────────────────────┤
│ VietNews     │ 1-doc    │ ~150K    │ SFT train, GRPO, Test  │
│ WikiLingua   │ 1-doc    │ ~19.5K   │ SFT train, GRPO, Test  │
│ ViMs         │ N-doc    │ 300 cls  │ GRPO train/val, Test   │
│ VLSP         │ N-doc    │ ~600     │ GRPO train/val, Test   │
├──────────────┴──────────┴──────────┴───────────────────────┤
│  Splits:                                                    │
│    SFT train:  VietNews/WL train (×3 aug)                   │
│    SFT val:    first 2K VN + 500 WL val (×3)                │
│    GRPO train: remaining val + VLSP train + ViMs 80%         │
│    GRPO val:   VLSP val + ViMs 20%                           │
│    Test:       first 2K VN + 500 WL + VLSP test+abmusu      │
├─────────────────────────────────────────────────────────────┤
│  Word counting: syllable-level (split by space)              │
│  Max source: 8000 chars | Max summary: 1500 chars            │
│  Underscore → space (preprocessing)                          │
└─────────────────────────────────────────────────────────────┘
```
