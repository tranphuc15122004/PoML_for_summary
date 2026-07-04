# Dữ liệu và chiến lược chia tập

**Cập nhật:** 03/07/2026  
**Code nguồn:** `src/dataset/dataset.py`, `src/dataset/augmenter.py`

## 1. Quy ước sử dụng trong báo cáo

- VietNews và WikiLingua là hai tập duy nhất dùng cho main generalization claims.
- ViMs và VLSP vẫn được giữ trong bảng kết quả nhưng phải gắn cảnh báo `exploratory` hoặc `leakage`.
- Aggregate `test.jsonl` với `N=3100` là artefact lịch sử hữu ích, nhưng không được dùng một mình để kết luận generalization.

## 2. Nguồn dữ liệu

| Dataset | Tác vụ | Reference | Quy mô raw dùng trong pipeline |
|---|---|---|---:|
| VietNews | Single-document news | Title | 105.418 train; 22.642 val |
| WikiLingua | Single-document how-to | Abstractive summary | 13.999 train; khoảng 1.680 val |
| ViMs | Multi-document news | Gold annotator 0 | 300 cluster |
| VLSP 2022 AbMuSu | Multi-document | Ưu tiên trường `summary`; fallback extractive label | 285 train; 15 val; 300 abmusu |

Mọi source bị truncate ở 8.000 ký tự và summary ở 1.500 ký tự; underscore segmentation marker được thay bằng khoảng trắng.

## 3. Xử lý từng dataset

### VietNews

- Target là dòng title đầu tiên.
- Source là sapo và body; code bỏ dòng cuối như caption.
- SFT chỉ giữ target có ít nhất 10 whitespace token.
- GRPO giữ toàn bộ mẫu, kể cả title ngắn.

### WikiLingua

- File JSONL có `src` và `tgt` dạng list câu.
- Loader nối mỗi list bằng khoảng trắng.
- Dùng cho cả SFT và GRPO.

### ViMs

- Mỗi cluster gồm nhiều tài liệu; source được ghép với header `[Tài liệu i]`.
- Dùng `0.gold.txt` làm reference mặc định.
- Shuffle seed 42, lấy 80% vào SFT/GRPO train và 20% vào GRPO val.
- Code test hiện lấy lại toàn bộ 300 cluster; đây là leakage, không phải held-out test.

### VLSP

- Source là nhiều document block.
- Target ưu tiên trường human-written `summary`; nếu thiếu mới dựng từ extractive labels.
- 285 train chỉ vào GRPO train; 15 val vào GRPO val.
- Test hiện dùng split `abmusu` 300 mẫu, trùng chính xác 285 train + 15 val.

## 4. Dữ liệu sinh hiện tại

| File | Số dòng | Nội dung |
|---|---:|---|
| `sft_train.jsonl` | 111.150 | Chat có assistant response |
| `sft_val.jsonl` | 2.500 | 2.000 VietNews + 500 WikiLingua |
| `grpo_train.jsonl` | 119.942 | Prompt, reference, constraint metadata |
| `grpo_val.jsonl` | 21.897 | Validation reward |
| `test_vietnews.jsonl` | 2.000 | Test single-doc |
| `test_wikilingua.jsonl` | 500 | Test single-doc |
| `test_vims.jsonl` | 300 | Exploratory; leaked |
| `test_vlsp.jsonl` | 300 | Exploratory; leaked |
| `test.jsonl` | 3.100 | Gộp bốn file trên |

### Thành phần train

| Nguồn | SFT train | GRPO train |
|---|---:|---:|
| VietNews | 96.911 | 105.418 |
| WikiLingua | 13.999 | 13.999 |
| ViMs | 240 | 240 |
| VLSP | 0 | 285 |
| **Tổng** | **111.150** | **119.942** |

## 5. Schema

### SFT

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Yêu cầu:\n- Độ dài: ...\n- Số câu: ...\n\nVăn bản:\n..."},
    {"role": "assistant", "content": "reference summary"}
  ],
  "meta": {
    "target_length": 22,
    "target_sentences": 1,
    "length_requirement": "khoảng 22 từ",
    "sentence_requirement": "khoảng 1 câu"
  }
}
```

SFT luôn dùng template `khoảng X` lấy trực tiếp từ reference.

### GRPO

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "reference": "reference summary",
  "meta": {
    "length_requirement": "trong khoảng 12-18 từ",
    "sentence_requirement": "trong khoảng 1-2 câu"
  }
}
```

Ba template được luân phiên theo counter để tạo đa dạng constraint.

## 6. Kiểm toán leakage

Exact overlap được tính bằng SHA-256 của source đã chuẩn hóa trong JSONL.

| Test | SFT train | SFT val | GRPO train | GRPO val |
|---|---:|---:|---:|---:|
| VietNews, N=2.000 | 7 | 0 | 8 | 2 |
| WikiLingua, N=500 | 0 | 0 | 0 | 0 |
| ViMs, N=300 | 240 | 0 | 240 | 60 |
| VLSP, N=300 | 0 | 0 | 285 | 15 |

Quy tắc sử dụng trong báo cáo:

- VietNews/WikiLingua: main results, nhưng ghi nhận duplicate nhỏ ở VietNews.
- ViMs/VLSP: chỉ exploratory/in-sample cho đến khi tái chia dữ liệu.
- Không dùng combined N=3.100 làm bằng chứng duy nhất về generalization.

## 7. Vấn đề thiết kế cần lưu ý

1. Constraint được suy ra từ reference, nên đây là reference-conditioned control benchmark; chưa kiểm tra constraint tùy ý ngoài phân bố gold.
2. SFT chỉ thấy `khoảng X`, trong khi GRPO thấy ba template; có distribution shift có chủ đích.
3. VietNews title là headline, không phải full abstractive summary.
4. Multi-document chỉ chiếm khoảng 0,2% train, nên khó kỳ vọng generalization mạnh.
5. Whitespace count không phải Vietnamese word segmentation chuẩn.

## 8. Mẫu cho pilot qualitative / human evaluation

Nếu thực hiện pilot human evaluation theo kế hoạch hiện tại:

- dùng seed `42` để lấy mẫu cố định;
- `15` mẫu VietNews;
- `15` mẫu WikiLingua;
- `5` mẫu ViMs;
- `5` mẫu VLSP.

ViMs/VLSP trong pilot này vẫn phải được gắn nhãn exploratory, không dùng để suy rộng chất lượng tổng quát.

## 9. Hướng cleanup về sau

- ViMs: chia cluster sạch theo train/val/test và lưu manifest hash.
- VLSP: xác định lại quan hệ giữa `abmusu` và train/val, sau đó tạo held-out IDs rõ ràng.
- VietNews: loại duplicate exact theo source hash giữa train/val/test.
- Lưu manifest ID/hash cho mọi split để audit tự động.
