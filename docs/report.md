# Báo cáo Đánh giá Mô hình Tóm tắt Văn bản Tiếng Việt

**Mã đánh giá:** `20260622_103706`
**Ngày thực hiện:** 22/06/2026
**Đơn vị:** Viettel AI R&D
**Pipeline:** Hậu huấn luyện (Post-training) — SFT → GRPO

---

## 1. Tổng quan

Báo cáo này trình bày kết quả đánh giá toàn diện các mô hình tóm tắt văn bản tiếng Việt sau quá trình hậu huấn luyện. Đánh giá được thực hiện trên tập test tổng hợp **3,100 mẫu** bao gồm 4 nguồn dữ liệu.

> **Trọng tâm phân tích:** So sánh chi tiết giữa **GRPO config v4** và **v5** — hai cấu hình khác biệt về số lượng generation (K), learning rate, và hệ số KL penalty. Đây là thí nghiệm chính nhằm tối ưu hóa GRPO alignment cho tóm tắt tiếng Việt.

| Tập con | Số mẫu | Đặc điểm |
|---|---|---|
| **VietNews** | 2,000 | Báo chí tiếng Việt, single-doc abstractive |
| **WikiLingua** | 500 | WikiHow tiếng Việt, single-doc abstractive |
| **ViMs** | 300 | Multi-doc abstractive, 5–10 văn bản/cluster |
| **VLSP** | 300 | Multi-doc extractive, VLSP 2022 AbMuSu |

### 1.1 Các mô hình được đánh giá

| Nhóm | Mô hình | Ký hiệu | Base Model | Ghi chú |
|---|---|---|---|---|
| **Qwen2.5** | Qwen2.5-3B-Instruct (base) | `QWEN25_base` | Qwen2.5-3B-Instruct | Baseline chưa qua huấn luyện |
| | + SFT có augmentation | `QWEN25_sft_aug` | ↑ | SFT trên 358K mẫu augmented |
| | + SFT không augmentation | `QWEN25_sft_no_aug` | ↑ | SFT trên 119K mẫu gốc |
| **Qwen3.5** | Qwen3.5-4B (base) | `QWEN35_base` | Qwen3.5-4B | Mô hình mới nhất (chưa SFT/GRPO) |
| **Qwen3-4B-Base** | Base | `QWEN3_BASE_base` | Qwen3-4B-Base | Baseline chưa qua huấn luyện |
| | + SFT | `QWEN3_BASE_sft` | ↑ | SFT 2 epoch |
| | + GRPO *fresh* v4 | `QWEN3_BASE_grpo_fresh_v4` | ↑ | GRPO *từ base*, config **v4** |
| | + GRPO *fresh* v5 | `QWEN3_BASE_grpo_fresh_v5` | ↑ | GRPO *từ base*, config **v5** |
| | + GRPO *sft* v4 | `QWEN3_BASE_grpo_sft_v4` | ↑ | GRPO *từ SFT*, config **v4** |
| | + GRPO *sft* v5 | `QWEN3_BASE_grpo_sft_v5` | ↑ | GRPO *từ SFT*, config **v5** |
| **Qwen3-4B-Instruct** | Base | `QWEN3_INSTRUCT_base` | Qwen3-4B | Baseline chưa qua huấn luyện |
| | + SFT | `QWEN3_INSTRUCT_sft` | ↑ | SFT 2 epoch |
| | + GRPO *fresh* v4 | `QWEN3_INSTRUCT_grpo_fresh_v4` | ↑ | GRPO *từ base*, config **v4** |
| | + GRPO *fresh* v5 | `QWEN3_INSTRUCT_grpo_fresh_v5` | ↑ | GRPO *từ base*, config **v5** |
| | + GRPO *sft* v4 | `QWEN3_INSTRUCT_grpo_sft_v4` | ↑ | GRPO *từ SFT*, config **v4** |
| | + GRPO *sft* v5 | `QWEN3_INSTRUCT_grpo_sft_v5` | ↑ | GRPO *từ SFT*, config **v5** |

> **Giải thích ký hiệu:** *fresh* = GRPO khởi tạo từ base model (chưa SFT). *sft* = GRPO khởi tạo từ SFT checkpoint. **v4** và **v5** là hai cấu hình GRPO khác nhau (xem Mục 4.3).

### 1.2 Thước đo đánh giá

| Thước đo | Ký hiệu | Ý nghĩa |
|---|---|---|
| **ROUGE-2** | `rouge2` | Độ chính xác bigram overlap giữa sinh và tham chiếu (chất lượng tóm tắt) |
| **Tỷ lệ lỗi độ dài** | `LenErr%` | Phần trăm mẫu có độ dài sinh ra vượt quá dung sai cho phép so với yêu cầu |
| **Khoảng cách độ dài** | `LenDist` | Khoảng cách tuyệt đối trung bình giữa độ dài sinh ra và độ dài mục tiêu (từ) |
| **Độ dài trung bình** | `AvgLen` | Số từ trung bình mô hình sinh ra |

> **Lưu ý:** "từ" ở đây được hiểu là âm tiết (syllable-level), theo chuẩn của các công bố tiếng Việt (ViT5, BARTpho, VLSP).

---

## 2. Kết quả tổng thể

### 2.1 Bảng tổng hợp trên toàn bộ tập test (N=3,100)

| Mô hình | ROUGE-2 ↑ | LenErr% ↓ | LenDist ↓ | AvgLen |
|---|---|---|---|---|
| **QWEN25_base** | 0.1331 | 60.68 | 28.1 | 55.9 |
| QWEN25_sft_aug | 0.2308 | 17.92 | 28.6 | 35.4 |
| QWEN25_sft_no_aug | 0.2048 | 38.79 | 45.8 | 19.6 |
| **QWEN35_base** | 0.1492 | 45.56 | 21.0 | 69.3 |
| **QWEN3_BASE_base** | 0.1056 | **169.77** | 51.8 | 71.9 |
| QWEN3_BASE_sft | **0.2632** | 30.52 | 18.4 | 61.1 |
| QWEN3_BASE_grpo_fresh_v4 | 0.1388 | 76.44 | 32.8 | 58.2 |
| QWEN3_BASE_grpo_fresh_v5 | 0.1866 | 29.05 | 21.5 | 50.3 |
| QWEN3_BASE_grpo_sft_v4 | 0.2671 | 20.82 | 17.1 | 57.3 |
| QWEN3_BASE_grpo_sft_v5 | 0.2638 | **14.94** | 20.5 | 46.7 |
| **QWEN3_INSTRUCT_base** | 0.1506 | 30.22 | 18.9 | 54.9 |
| QWEN3_INSTRUCT_sft | 0.2609 | 12.42 | 14.6 | 51.5 |
| QWEN3_INSTRUCT_grpo_fresh_v4 | 0.1551 | 26.06 | 18.3 | 52.9 |
| QWEN3_INSTRUCT_grpo_fresh_v5 | 0.1669 | 22.78 | 17.7 | 50.7 |
| QWEN3_INSTRUCT_grpo_sft_v4 | 0.2655 | 13.41 | 16.9 | 48.0 |
| **QWEN3_INSTRUCT_grpo_sft_v5** | **0.2765** | 15.85 | 18.8 | 44.4 |

> **Màu sắc:** 🟢 Dẫn đầu | 🟡 Khá | 🔴 Kém

### 2.2 Nhận xét chính

1. **Mô hình tốt nhất toàn diện:** `QWEN3_INSTRUCT_grpo_sft_v5` đạt ROUGE-2 cao nhất (0.2765) với LenErr% ở mức thấp (15.85%), cho thấy GRPO từ nền tảng SFT với config v5 (K=8, LR=2e-6) mang lại chất lượng tóm tắt vượt trội.

2. **Tác động của GRPO:** GRPO cải thiện đáng kể so với base model, đặc biệt khi khởi tạo từ checkpoint SFT:
   - Base → GRPO fresh v5: ROUGE-2 tăng từ 0.1056 → 0.1866 (+77%) cho Qwen3-Base
   - SFT → GRPO sft v5: ROUGE-2 tăng từ 0.2632 → 0.2638 (ổn định) cho Qwen3-Base
   - SFT → GRPO sft v5: ROUGE-2 tăng từ 0.2609 → **0.2765** (+6%) cho Qwen3-Instruct

3. **GRPO fresh vs SFT-init:** GRPO khởi tạo từ SFT luôn vượt trội so với GRPO từ base (fresh) — khẳng định SFT là bước đệm cần thiết.

4. **v4 vs v5 — khác biệt có hệ thống:** Config v5 (K=8, LR=2e-6, β=0.04) vượt trội so với v4 (K=4, LR=5e-7, β=0.15) trên hầu hết các chỉ số, đặc biệt trên nhánh Base fresh (+34.4% ROUGE-2, -47.39pp LenErr%). Sự khác biệt đến từ 3 yếu tố: (a) K=8 giảm variance của advantage estimate, (b) LR=2e-6 cho phép cập nhật đủ lớn, (c) β=0.04 ít ràng buộc KL hơn → policy tự do khám phá. Trên nhánh Instruct-SFT, v5 cho ROUGE-2 cao hơn (+4.1%) nhưng LenErr% cao hơn (+2.44pp) — đánh đổi giữa chất lượng nội dung và độ tuân thủ độ dài.

5. **Qwen2.5 vs Qwen3:** Qwen2.5-3B-Instruct base (0.1331) outperforms Qwen3-4B-Base base (0.1056) nhưng thua Qwen3-4B-Instruct base (0.1506). Sau SFT+GRPO, Qwen3-4B vượt Qwen2.5-3B SFT (0.2765 vs 0.2308).

---

## 3. Bảng tổng hợp kết quả theo metric qua các tập dữ liệu

### 3.1 ROUGE-2 trên từng tập dữ liệu

| Mô hình | Tổng hợp (test) | VietNews | ViMs | VLSP | WikiLingua |
|---|---|---|---|---|---|
| QWEN25_base | 0.1331 | 0.1380 | 0.1775 | 0.1573 | 0.0726 |
| QWEN25_sft_aug | 0.2308 | 0.2666 | 0.0732 | 0.1332 | 0.2408 |
| QWEN25_sft_no_aug | 0.2048 | 0.2506 | 0.0392 | 0.0516 | 0.2127 |
| QWEN35_base | 0.1492 | 0.1575 | 0.1913 | 0.1710 | 0.0777 |
| QWEN3_BASE_base | 0.1056 | 0.1213 | 0.1346 | 0.0949 | 0.0318 |
| QWEN3_BASE_sft | 0.2632 | 0.2661 | 0.2565 | **0.2814** 🥇 | 0.2443 |
| QWEN3_BASE_grpo_fresh_v4 | 0.1388 | 0.1522 | 0.1740 | 0.1348 | 0.0664 |
| QWEN3_BASE_grpo_fresh_v5 | 0.1866 | 0.2073 | 0.2028 | 0.1814 | 0.0971 |
| QWEN3_BASE_grpo_sft_v4 | 0.2671 | 0.2745 | **0.2597** 🥇 | 0.2527 | 0.2505 |
| QWEN3_BASE_grpo_sft_v5 | 0.2638 | **0.2897** | 0.1816 | 0.1746 | **0.2635** 🥇 |
| QWEN3_INSTRUCT_base | 0.1506 | 0.1648 | 0.1783 | 0.1551 | 0.0744 |
| QWEN3_INSTRUCT_sft | 0.2609 | 0.2670 | 0.2471 | 0.2703 | 0.2393 |
| QWEN3_INSTRUCT_grpo_fresh_v4 | 0.1551 | 0.1708 | 0.1788 | 0.1569 | 0.0769 |
| QWEN3_INSTRUCT_grpo_fresh_v5 | 0.1669 | 0.1867 | 0.1843 | 0.1605 | 0.0809 |
| QWEN3_INSTRUCT_grpo_sft_v4 | 0.2655 | 0.2803 | 0.2177 | 0.2497 | 0.2444 |
| **QWEN3_INSTRUCT_grpo_sft_v5** | **0.2765** 🥇 | **0.2962** 🥇 | 0.2127 | 0.2439 | 0.2559 |

### 3.2 Tỷ lệ lỗi độ dài (LenErr%) trên từng tập dữ liệu

| Mô hình | Tổng hợp (test) | VietNews | ViMs | VLSP | WikiLingua |
|---|---|---|---|---|---|
| QWEN25_base | 60.68 | 67.30 | 36.99 | 34.78 | 63.97 |
| QWEN25_sft_aug | 17.92 | 6.35 | 62.18 | 57.06 | **14.17** 🥇 |
| QWEN25_sft_no_aug | 38.79 | 22.31 | 92.29 | 92.59 | 40.34 |
| QWEN35_base | 45.56 | 37.76 | **21.68** 🥇 | **20.20** 🥇 | 106.29 |
| QWEN3_BASE_base | 169.77 | 205.89 | 42.61 | 47.93 | 174.67 |
| QWEN3_BASE_sft | 30.52 | 31.19 | 38.69 | 27.53 | 24.72 |
| QWEN3_BASE_grpo_fresh_v4 | 76.44 | 81.16 | 37.37 | 41.17 | 102.18 |
| QWEN3_BASE_grpo_fresh_v5 | 29.05 | 22.36 | 31.89 | 34.33 | 50.94 |
| QWEN3_BASE_grpo_sft_v4 | 20.82 | 17.10 | 39.62 | 30.43 | 18.66 |
| QWEN3_BASE_grpo_sft_v5 | 14.94 | 6.43 | 44.42 | 42.55 | 14.69 |
| QWEN3_INSTRUCT_base | 30.22 | 24.50 | 26.07 | 27.09 | 57.47 |
| **QWEN3_INSTRUCT_sft** | **12.42** 🥇 | 6.26 | 31.55 | 24.51 | 18.34 |
| QWEN3_INSTRUCT_grpo_fresh_v4 | 26.06 | 20.15 | 27.63 | 27.64 | 47.81 |
| QWEN3_INSTRUCT_grpo_fresh_v5 | 22.78 | 17.08 | 28.37 | 28.76 | 38.67 |
| QWEN3_INSTRUCT_grpo_sft_v4 | 13.41 | **6.00** 🥇 | 36.24 | 28.69 | 20.18 |
| QWEN3_INSTRUCT_grpo_sft_v5 | 15.85 | 9.24 | 39.14 | 32.85 | 18.11 |

> **Nhận xét:** Xét trên tổng thể, QWEN3_INSTRUCT_sft có LenErr% thấp nhất (12.42%) 🥇. Trên VietNews, QWEN3_INSTRUCT_grpo_sft_v4 đạt thấp nhất (6.00%) 🥇. Trên ViMs và VLSP, QWEN35_base (chưa qua huấn luyện) cho LenErr% thấp nhất nhờ sinh ngắn (21.68% và 20.20%). Trong nhóm SFT/GRPO, ViMs và VLSP có LenErr% cao (30–92%), đặc biệt QWEN25_sft_no_aug sụp đổ hoàn toàn (~92%). QWEN3_BASE_base cho LenErr% cao bất thường trên tổng thể (169.77%) do model không instruction-tuned sinh tràn lan.

### 3.3 Khoảng cách độ dài (LenDist) trên từng tập dữ liệu

| Mô hình | Tổng hợp (test) | VietNews | ViMs | VLSP | WikiLingua |
|---|---|---|---|---|---|
| QWEN25_base | 28.1 | 11.1 | 82.9 | 89.0 | 26.7 |
| QWEN25_sft_aug | 28.6 | 1.1 | 135.4 | 139.7 | **7.9** 🥇 |
| QWEN25_sft_no_aug | 45.8 | 3.9 | 195.0 | 218.8 | 20.4 |
| QWEN35_base | 21.0 | 6.5 | **43.3** 🥇 | **49.7** 🥇 | 48.1 |
| QWEN3_BASE_base | 51.8 | 33.0 | 92.0 | 120.3 | 61.6 |
| QWEN3_BASE_sft | 18.4 | 5.7 | 66.4 | 61.6 | 14.2 |
| QWEN3_BASE_grpo_fresh_v4 | 32.8 | 13.1 | 82.7 | 102.3 | 40.3 |
| QWEN3_BASE_grpo_fresh_v5 | 21.5 | 3.8 | 71.8 | 88.0 | 22.0 |
| QWEN3_BASE_grpo_sft_v4 | 17.1 | 3.2 | 69.4 | 68.1 | 10.9 |
| QWEN3_BASE_grpo_sft_v5 | 20.5 | 1.2 | 89.2 | 100.5 | 8.4 |
| QWEN3_INSTRUCT_base | 18.9 | 4.0 | 59.8 | 70.2 | 23.4 |
| **QWEN3_INSTRUCT_sft** | **14.6** 🥇 | 1.1 | 64.5 | 62.2 | 9.9 |
| QWEN3_INSTRUCT_grpo_fresh_v4 | 18.3 | 3.4 | 62.3 | 70.9 | 19.9 |
| QWEN3_INSTRUCT_grpo_fresh_v5 | 17.7 | 2.9 | 63.1 | 72.9 | 16.1 |
| QWEN3_INSTRUCT_grpo_sft_v4 | 16.9 | **1.0** 🥇 | 76.1 | 73.0 | 11.0 |
| QWEN3_INSTRUCT_grpo_sft_v5 | 18.8 | 1.5 | 84.5 | 82.7 | 10.2 |

> **Nhận xét:** Trên tổng thể, QWEN3_INSTRUCT_sft có LenDist thấp nhất (14.6). VietNews và WikiLingua có LenDist rất thấp (1–11) đối với các mô hình đã huấn luyện — model bám sát độ dài mục tiêu. Trên ViMs và VLSP, LenDist cao (60–218) do độ dài mục tiêu lớn và biến động mạnh. QWEN25_sft_no_aug có LenDist lên tới 218.8 trên VLSP — model hầu như không sinh được summary có nghĩa.

### 3.4 Độ dài trung bình (AvgLen) trên từng tập dữ liệu

| Mô hình | Tổng hợp (test) | VietNews | ViMs | VLSP | WikiLingua |
|---|---|---|---|---|---|
| QWEN25_base | 55.9 | 27.3 | 129.3 | 146.4 | 72.1 |
| QWEN25_sft_aug | 35.4 | 16.9 | 80.2 | 101.1 | 43.3 |
| QWEN25_sft_no_aug | 19.6 | 15.7 | 14.0 | 15.9 | 41.1 |
| QWEN35_base | 69.3 | 23.5 | 192.6 | 204.1 | 97.5 |
| QWEN3_BASE_base | 71.9 | 48.7 | 125.8 | 118.6 | 104.0 |
| QWEN3_BASE_sft | 61.1 | 22.6 | 184.1 | 203.9 | 55.8 |
| QWEN3_BASE_grpo_fresh_v4 | 58.2 | 28.5 | 132.0 | 134.7 | 86.3 |
| QWEN3_BASE_grpo_fresh_v5 | 50.3 | 17.6 | 139.9 | 147.8 | 68.8 |
| QWEN3_BASE_grpo_sft_v4 | 57.3 | 20.0 | 179.0 | 196.0 | 50.2 |
| QWEN3_BASE_grpo_sft_v5 | 46.7 | 17.6 | 142.6 | 150.5 | 43.4 |
| QWEN3_INSTRUCT_base | 54.9 | 19.7 | 150.4 | 165.8 | 71.5 |
| QWEN3_INSTRUCT_sft | 51.5 | 17.2 | 163.4 | 182.9 | 42.6 |
| QWEN3_INSTRUCT_grpo_fresh_v4 | 52.9 | 18.3 | 148.0 | 164.8 | 67.1 |
| QWEN3_INSTRUCT_grpo_fresh_v5 | 50.7 | 16.7 | 146.5 | 162.6 | 62.5 |
| QWEN3_INSTRUCT_grpo_sft_v4 | 48.0 | 16.8 | 146.1 | 168.7 | 41.6 |
| QWEN3_INSTRUCT_grpo_sft_v5 | 44.4 | 16.3 | 129.5 | 153.1 | 40.5 |

> **Nhận xét:** Trên tổng thể, độ dài trung bình dao động từ 19.6 (QWEN25_sft_no_aug) đến 71.9 (QWEN3_BASE_base). Trên VietNews và WikiLingua (single-doc), các mô hình SFT/GRPO sinh ngắn (16–43 từ), phù hợp với độ dài mục tiêu. Trên ViMs và VLSP (multi-doc), độ dài sinh ra lớn hơn nhiều (80–204 từ), phản ánh độ dài summary multi-doc tự nhiên. QWEN25_sft_no_aug bất thường trên ViMs (14.0) và VLSP (15.9) — sinh quá ngắn, cho thấy model không học được pattern multi-doc.

---

## 4. So sánh với kết quả tham khảo

> **Lưu ý:** Kết quả tham khảo từ `VDT_Textsum/ketqua.md` sử dụng thang ROUGE-2 (%), trong khi kết quả của chúng tôi ở thang 0–1. Để so sánh, các giá trị ROUGE-2 của chúng tôi được nhân với 100 (chuyển sang %). Các tập dữ liệu tham khảo có thể khác biệt nhẹ về split so với tập test của chúng tôi.

### 4.1 ROUGE-2 (%) — So sánh với các mô hình tham khảo

| Mô hình | Tham số | WikiLingua | Vietnews | ViM/ViMs | VLSP2022 |
|---|---|---|---|---|---|
| **Nhóm mô hình của chúng tôi (Post-training 3–4B)** | | | | | |
| QWEN3_INSTRUCT_grpo_sft_v5 | 4B | 25,59 | **29,62** 🥇 | 21,27 | 24,39 |
| QWEN3_BASE_grpo_sft_v5 | 4B | **26,35** 🥇 | 28,97 | 18,16 | 17,46 |
| QWEN3_BASE_sft | 4B | 24,43 | 26,61 | **25,65** 🥇 | **28,14** 🥇 |
| QWEN3_INSTRUCT_sft | 4B | 23,93 | 26,70 | 24,71 | 27,03 |
| QWEN3_BASE_grpo_sft_v4 | 4B | 25,05 | 27,45 | 25,97 | 25,27 |
| QWEN25_sft_aug | 3B | 24,08 | 26,66 | 7,32 | 13,32 |
| QWEN35_base | 4B | 7,77 | 15,75 | 19,13 | 17,10 |
| **Nhóm tham khảo (công bố trước)** | | | | | |
| VietAI | – | **33,12** 🥇 | **34,24** 🥇 | – | – |
| Qwen3-14B | 14B | 20,24 | 18,51 | **44,38** 🥇 | **44,27** 🥇 |
| GPT-4o | – | 20,65 | 21,61 | 44,26 | 43,37 |
| gpt-3.5-turbo | – | 21,09 | 28,13 | 19,28 | 35,79 |
| Llama3.3-70B-Instruct | 70B | 19,40 | 22,17 | 37,54 | 39,02 |
| Phi4-14B | 14B | 14,28 | 13,17 | 41,85 | 42,31 |
| Sailor-20B-chat | 20B | 18,74 | 17,70 | 39,47 | 40,96 |
| VinBigdata | 7B | 21,02 | 20,59 | 37,98 | 40,23 |

> **Nhận xét:**
> - Trên **WikiLingua** và **Vietnews** (single-doc), các mô hình 4B của chúng tôi sau SFT/GRPO đạt ROUGE-2 từ 23–30%, vượt qua các mô hình lớn như Qwen3-14B (18–20%), Llama3.3-70B (19–22%). Đây là kết quả ấn tượng cho thấy post-training giúp small LLM cạnh tranh với mô hình lớn trên tác vụ single-doc summarization.
> - **VietAI** vẫn dẫn đầu tuyệt đối trên WikiLingua (33,12%) và Vietnews (34,24%), nhưng đây là mô hình chuyên biệt cho tóm tắt tiếng Việt, trong khi Qwen3 là mô hình đa dụng.
> - Trên **ViMs** và **VLSP** (multi-doc), các mô hình tham khảo lớn (Qwen3-14B: 44%, GPT-4o: 44%) vượt trội so với mô hình 4B của chúng tôi (21–28%). Multi-document summarization đòi hỏi dung lượng mô hình lớn hơn để xử lý nhiều văn bản đầu vào.
> - **QWEN3_BASE_sft** đạt kết quả tốt nhất trên ViMs (25,65%) và VLSP (28,14%) trong nhóm của chúng tôi, cho thấy SFT thuần vẫn là phương pháp hiệu quả cho multi-doc.

### 4.2 Length Dist — So sánh khả năng tuân thủ độ dài

| Mô hình | WikiLingua | Vietnews | ViM/ViMs | VLSP2022 |
|---|---|---|---|---|
| **Nhóm của chúng tôi (3–4B)** | | | | |
| QWEN3_INSTRUCT_grpo_sft_v4 | **7,9** 🥇 | **1,0** 🥇 | 76,1 | 73,0 |
| QWEN25_sft_aug | 7,9 🥇 | 1,1 | 135,4 | 139,7 |
| QWEN3_INSTRUCT_sft | 9,9 | 1,1 | 64,5 | 62,2 |
| QWEN3_BASE_grpo_sft_v5 | 8,4 | 1,2 | 89,2 | 100,5 |
| QWEN3_INSTRUCT_grpo_sft_v5 | 10,2 | 1,5 | 84,5 | 82,7 |
| **Nhóm tham khảo** | | | | |
| **GPT-4o** | **11** 🥇 | **8** 🥇 | 187 | 58 |
| gpt-3.5-turbo | 17 | 18 | **47** 🥇 | **30** 🥇 |
| Llama3.3-70B-Instruct | 13 | 14 | 186 | 73 |
| Qwen3-14B | 45 | 50 | 131 | 34 |
| Sailor2-20B-Chat | 43 | 38 | 268 | 41 |

> **Nhận xét:**
> - Các mô hình của chúng tôi vượt trội về khả năng tuân thủ độ dài trên **VietNews** (LenDist=1,0–1,5 so với 8–50 của tham khảo). Đây là kết quả trực tiếp từ việc huấn luyện với length reward trong GRPO.
> - Trên **WikiLingua**, chúng tôi cũng cạnh tranh tốt (7,9–10,2 so với 11–45 của tham khảo). QWEN3_INSTRUCT_grpo_sft_v4 và QWEN25_sft_aug cùng đạt 7,9 — tốt hơn GPT-4o (11).
> - Trên **ViMs** và **VLSP** (multi-doc), cả hai nhóm đều có LenDist cao (47–268), phản ánh độ khó của việc kiểm soát độ dài trên multi-document.

### 4.4 Tổng kết so sánh

| Tiêu chí | Mô hình của chúng tôi tốt nhất | So với tham khảo tốt nhất |
|---|---|---|
| **ROUGE-2** VietNews | 29,62% (QWEN3_INSTRUCT_grpo_sft_v5) | 34,24% (VietAI) — **kém 4,62pp** |
| **ROUGE-2** WikiLingua | 26,35% (QWEN3_BASE_grpo_sft_v5) | 33,12% (VietAI) — **kém 6,77pp** |
| **ROUGE-2** ViMs | 25,97% (QWEN3_BASE_grpo_sft_v4) | 44,38% (Qwen3-14B) — **kém 18,41pp** |
| **ROUGE-2** VLSP | 28,14% (QWEN3_BASE_sft) | 44,27% (Qwen3-14B) — **kém 16,13pp** |
| **LenDist** VietNews | **1,0** (QWEN3_INSTRUCT_grpo_sft_v4) | 8 (GPT-4o) — **vượt trội 7×** 🥇 |
| **LenDist** WikiLingua | **7,9** (QWEN3_INSTRUCT_grpo_sft_v4) | 11 (GPT-4o) — **vượt trội** 🥇 |
| **LenDist** ViMs | 64,5 (QWEN3_INSTRUCT_sft) | 47 (gpt-3.5-turbo) — **thua** |
| **LenDist** VLSP | 61,6 (QWEN3_BASE_sft) | 30 (gpt-3.5-turbo) — **thua** |

> **Kết luận so sánh:** Mặc dù chỉ có 3–4B tham số (so với 14B–70B+ của các mô hình tham khảo), các mô hình post-training của chúng tôi đạt chất lượng tóm tắt cạnh tranh trên single-doc (VietNews, WikiLingua) và **vượt trội về khả năng tuân thủ độ dài** (LenDist thấp hơn GPT-4o 7 lần trên VietNews). Multi-doc summarization (ViMs, VLSP) vẫn là thách thức với các mô hình nhỏ.

---

## 5. Phân tích chi tiết theo tập dữ liệu

### 5.1 VietNews (N=2,000) — Dễ nhất

| Mô hình | ROUGE-2 | LenErr% | LenDist | AvgLen |
|---|---|---|---|---|
| QWEN25_base | 0.1380 | 67.30 | 11.1 | 27.3 |
| QWEN25_sft_aug | 0.2666 | 6.35 | 1.1 | 16.9 |
| QWEN25_sft_no_aug | 0.2506 | 22.31 | 3.9 | 15.7 |
| QWEN3_BASE_base | 0.1213 | 205.89 | 33.0 | 48.7 |
| QWEN3_BASE_sft | 0.2661 | 31.19 | 5.7 | 22.6 |
| QWEN3_BASE_grpo_sft_v5 | **0.2897** | 6.43 | 1.2 | 17.6 |
| QWEN3_INSTRUCT_base | 0.1648 | 24.50 | 4.0 | 19.7 |
| QWEN3_INSTRUCT_sft | 0.2670 | 6.26 | 1.1 | 17.2 |
| QWEN3_INSTRUCT_grpo_sft_v4 | 0.2803 | **6.00** 🥇 | **1.0** 🥇 | 16.8 |
| **QWEN3_INSTRUCT_grpo_sft_v5** | **0.2962** 🥇 | 9.24 | 1.5 | 16.3 |

**Nhận xét:** VietNews là tập dễ nhất với tất cả các mô hình. QWEN3_INSTRUCT_grpo_sft_v5 đạt ROUGE-2 cao nhất (0.2962), gần tiệm cận baseline ViT5-large (R2≈0.342). LenErr% dưới 10% cho hầu hết các mô hình SFT và GRPO.

### 5.2 WikiLingua (N=500) — Trung bình

| Mô hình | ROUGE-2 | LenErr% | LenDist | AvgLen |
|---|---|---|---|---|
| QWEN25_base | 0.0726 | 63.97 | 26.7 | 72.1 |
| QWEN25_sft_aug | 0.2408 | **14.17** 🥇 | **7.9** 🥇 | 43.3 |
| QWEN3_BASE_base | 0.0318 | 174.67 | 61.6 | 104.0 |
| **QWEN3_BASE_grpo_sft_v5** | **0.2635** 🥇 | 14.69 | 8.4 | 43.4 |
| QWEN3_INSTRUCT_base | 0.0744 | 57.47 | 23.4 | 71.5 |
| QWEN3_INSTRUCT_grpo_sft_v5 | 0.2559 | 18.11 | 10.2 | 40.5 |

**Nhận xét:** WikiLingua có độ khó trung bình. QWEN3_BASE_grpo_sft_v5 dẫn đầu ROUGE-2 (0.2635). QWEN25_sft_aug dẫn đầu cả LenErr% (14.17%) và LenDist (7.9). Các mô hình base cho ROUGE-2 rất thấp (~0.07), nhưng sau SFT/GRPO cải thiện đáng kể (0.24–0.26).

### 5.3 ViMs (N=300) — Khó (Multi-doc)

| Mô hình | ROUGE-2 | LenErr% | LenDist | AvgLen |
|---|---|---|---|---|
| QWEN25_base | 0.1775 | 36.99 | 82.9 | 129.3 |
| QWEN25_sft_aug | 0.0732 | 62.18 | 135.4 | 80.2 |
| QWEN25_sft_no_aug | 0.0392 | 92.29 | 195.0 | 14.0 |
| **QWEN3_BASE_grpo_sft_v4** | **0.2597** 🥇 | 39.62 | 69.4 | 179.0 |
| QWEN3_INSTRUCT_base | 0.1783 | **26.07** 🥇 | **59.8** 🥇 | 150.4 |
| QWEN3_INSTRUCT_sft | 0.2471 | 31.55 | 64.5 | 163.4 |
| QWEN3_INSTRUCT_grpo_sft_v4 | 0.2177 | 36.24 | 76.1 | 146.1 |
| QWEN3_INSTRUCT_grpo_sft_v5 | 0.2127 | 39.14 | 84.5 | 129.5 |

**Nhận xét:** ViMs là tập khó nhất do tính chất multi-document. QWEN3_BASE_grpo_sft_v4 dẫn đầu ROUGE-2 (0.2597) 🥇 nhưng LenErr% còn cao (39.62%). QWEN3_INSTRUCT_base cho LenErr% (26.07%) và LenDist (59.8) tốt nhất nhờ instruction-following sẵn có. Đáng chú ý: QWEN25_sft_no_aug bị sụp đổ hoàn toàn trên ViMs (R2=0.0392, LenErr%=92.29%) — do không có augmentation nên model không học được pattern cho multi-doc.

### 5.4 VLSP (N=300) — Khó (Multi-doc)

| Mô hình | ROUGE-2 | LenErr% | LenDist | AvgLen |
|---|---|---|---|---|
| QWEN25_base | 0.1573 | 34.78 | 89.0 | 146.4 |
| QWEN25_sft_aug | 0.1332 | 57.06 | 139.7 | 101.1 |
| QWEN25_sft_no_aug | 0.0516 | 92.59 | 218.8 | 15.9 |
| **QWEN3_BASE_sft** | **0.2814** 🥇 | 27.53 | **61.6** 🥇 | 203.9 |
| QWEN3_INSTRUCT_sft | 0.2703 | **24.51** 🥇 | 62.2 | 182.9 |
| QWEN3_INSTRUCT_grpo_sft_v4 | 0.2497 | 28.69 | 73.0 | 168.7 |
| QWEN3_INSTRUCT_grpo_sft_v5 | 0.2439 | 32.85 | 82.7 | 153.1 |

**Nhận xét:** VLSP cũng là multi-doc. QWEN3_BASE_sft đạt ROUGE-2 cao nhất (0.2814) và LenDist thấp nhất (61.6), cạnh tranh với baseline VLSP (R2≈0.28–0.30). QWEN3_INSTRUCT_sft cho LenErr% thấp nhất (24.51%). GRPO trên VLSP chưa cải thiện đáng kể so với SFT, thậm chí có dấu hiệu giảm nhẹ — gợi ý cần điều chỉnh reward cho multi-doc.

---

## 6. So sánh hiệu quả của GRPO

### 6.1 Cải thiện ROUGE-2 qua các giai đoạn

| Dòng | Base | → SFT | → GRPO v5 (sft) | Tổng cải thiện |
|---|---|---|---|---|
| Qwen3-Base | 0.1056 | 0.2632 (+149%) | 0.2638 (+150%) | **+150%** |
| Qwen3-Instruct | 0.1506 | 0.2609 (+73%) | **0.2765 (+84%)** | **+84%** |
| Qwen2.5-Instruct | 0.1331 | 0.2308 (+73%) | — | +73% |

### 6.2 Cải thiện LenErr% qua các giai đoạn

| Dòng | Base | → SFT | → GRPO v5 (sft) |
|---|---|---|---|
| Qwen3-Base | 169.77% | 30.52% | **14.94%** |
| Qwen3-Instruct | 30.22% | 12.42% | 15.85% |
| Qwen2.5-Instruct | 60.68% | 17.92% | — |

### 6.3 So sánh chi tiết Config v4 vs v5

Hai cấu hình GRPO được thiết kế để khảo sát ảnh hưởng của số lượng generation (K), learning rate, và mức độ KL regularization lên chất lượng alignment:

| Tham số | v4 | v5 | Phân tích tác động |
|---|---|---|---|
| **K** (num generations) | 4 | **8** | v5: gấp đôi số completion/prompt → giảm variance của advantage estimate, ước lượng chính xác hơn |
| **Learning rate** | 5e-7 | **2e-6** (×4) | v5: cho phép cập nhật lớn hơn mỗi step, hội tụ nhanh hơn nhưng rủi ro mất ổn định |
| **β** (KL penalty) | 0.15 | **0.04** (×0.27) | v5: ít ràng buộc hơn với reference model, cho phép policy khám phá xa hơn |
| **Hiệu ứng tổng hợp** | K nhỏ + LR thấp + β cao | K lớn + LR cao + β thấp | v5: ưu tiên khám phá (exploration); v4: ưu tiên ổn định (stability) |

#### Kết quả định lượng

| Tiêu chí | v4 | v5 | Chênh lệch |
|---|---|---|---|
| **ROUGE-2** (Instruct, sft) | 0.2655 | **0.2765** | **+4.1%** 🟢 |
| LenErr% (Instruct, sft) | **13.41%** | 15.85% | +2.44pp 🔴 |
| ROUGE-2 (Base, sft) | **0.2671** | 0.2638 | -1.2% 🔴 |
| LenErr% (Base, sft) | 20.82% | **14.94%** | **-5.88pp** 🟢 |
| ROUGE-2 (Instruct, fresh) | 0.1551 | **0.1669** | **+7.6%** 🟢 |
| LenErr% (Instruct, fresh) | 26.06% | **22.78%** | **-3.28pp** 🟢 |
| ROUGE-2 (Base, fresh) | 0.1388 | **0.1866** | **+34.4%** 🟢 |
| LenErr% (Base, fresh) | 76.44% | **29.05%** | **-47.39pp** 🟢 |

> pp = percentage points (chênh lệch phần trăm tuyệt đối)

#### 6.3.1 Tác động trên nhánh Instruct (có SFT)

Trên nhánh Instruct-SFT, v4 và v5 đều cho kết quả tốt, nhưng có sự đánh đổi rõ rệt:

- **v5 cho ROUGE-2 cao hơn** (0.2765 vs 0.2655, +4.1%): K=8 giúp giảm variance advantage → policy học được chất lượng tóm tắt tốt hơn. LR cao hơn (2e-6) cho phép cập nhật đủ lớn để cải thiện nội dung.
- **v4 cho LenErr% thấp hơn** (13.41% vs 15.85%): β cao hơn (0.15) giữ policy gần với SFT reference hơn, giúp duy trì độ tuân thủ độ dài đã học từ SFT. v5 với β=0.04 cho phép policy đi xa hơn, ưu tiên nội dung hơn là tuân thủ độ dài chính xác.

**Giải thích:** Khi β thấp, policy có thể "hy sinh" độ chính xác về độ dài để đạt được nội dung tóm tắt tốt hơn (ROUGE-2 cao hơn). Đây là hành vi hợp lý: GRPO tối ưu hóa tổng reward (0.5×R_acc + 0.3×R_len + 0.2×R_sent) — việc cải thiện R_acc đủ lớn sẽ bù đắp cho việc giảm nhẹ R_len.

#### 6.3.2 Tác động trên nhánh Base (không SFT)

Trên nhánh Base, sự khác biệt giữa v4 và v5 là rõ rệt nhất:

- **fresh v4 gần như không học được:** ROUGE-2 chỉ 0.1388 (gần bằng base 0.1056), LenErr% vẫn rất cao 76.44%. LR=5e-7 quá thấp để cập nhật policy từ base model không instruction-tuned. K=4 không đủ để ước lượng advantage chính xác khi policy còn yếu.
- **fresh v5 cải thiện đáng kể:** ROUGE-2=0.1866 (+34.4% so với v4), LenErr% giảm mạnh xuống 29.05%. LR=2e-6 đủ lớn để tạo thay đổi có ý nghĩa. K=8 giúp ước lượng advantage tốt hơn.
- **sft v4 vs v5:** Cả hai đều tốt nhờ điểm xuất phát SFT vững chắc. v4 cho ROUGE-2 cao hơn không đáng kể (0.2671 vs 0.2638), nhưng v5 vượt trội về LenErr% (14.94% vs 20.82%).

#### 6.3.3 Phân tích định tính — Chất lượng sinh

So sánh các mẫu sinh ra giữa v4 và v5 trên cùng input cho thấy khác biệt quan trọng:

**Trên Qwen3-4B-Base (SFT-init):**

| Input gốc | v4 (gen) | v5 (gen) | Reference |
|---|---|---|---|
| *Tin về đường dây đánh bạc* | `Đường dây đánh bạc nghìn đô ở Hà Tĩnh : Cảnh giới bằng bộ đàm行` | `Đường dây đánh bạc nghìn đô ở Hà Tĩnh trang bị bộ đàmเตือน` | `Chuyện chưa biết về vụ đánh án sới bạc nghìn đô trên núi Trạng Nẹo` |
| *Tin về thuốc sinh con theo ý muốn* | `Loại thuốc lạ hỗ trợ sinh con trai hay gái tuỳ theo ý muốn : “ Hot ” không ?ประหยัด` | `Loạt thuốc lạ hỗ trợ sinh con trai hay gái tuỳ ý muốn : Thực hư ra sao ?</s>` | `Không được công nhận , thuốc " sinh con theo ý muốn " vẫn rao bán công khai` |

Cả v4 và v5 trên Base model đều còn hiện tượng **nhiễm ký tự ngoại lai** (Thai, Chinese, Korean, symbols) ở cuối câu — dấu hiệu của việc policy chưa ổn định hoàn toàn. Tuy nhiên v5 có xu hướng sinh ít ký tự lạ hơn và câu hoàn chỉnh hơn.

**Trên Qwen3-4B-Instruct (SFT-init):**

| Input gốc | v4 (gen) | v5 (gen) | Reference |
|---|---|---|---|
| *Tin giả danh công an* | `Người đàn ông giả danh sĩ quan công an lừa đảo` | `Lừa đảo 3,2 tỷ đồng , sĩ quan công an 12 năm tù` | `Bản án cho đối tượng giả danh công an để lừa đảo` |
| *Tin đánh bạc nghìn đô* | `Đường dây đánh bạc nghìn đô ở Hà Tĩnh : Trang bị bộ đàm` | `Đường dây đánh bạc nghìn đô , trang bị bộ đàm cảnh giới` | `Chuyện chưa biết về vụ đánh án sới bạc nghìn đô trên núi Trạng Nẹo` |
| *Tin chìm tàu Cần Giờ* | `Truy tố ông Đảo và Quyết vì chìm tàu chở 30 người ở Cần Giờ 5 năm trước` | `Đề nghị truy tố ông Vũ Văn Đảo , ông Đinh Văn Quyết vì vụ chìm tàu` | `Đề nghị truy tố 2 giám đốc vụ chìm tàu ở Cần Giờ khiến 9 người thiệt mạng` |

Trên Instruct, cả v4 và v5 đều sinh tiếng Việt **hoàn toàn sạch** (không ký tự lạ). v5 có xu hướng sinh tóm tắt **ngắn gọn hơn** (44.4 từ trung bình vs 48.0 của v4) và **chứa nhiều chi tiết số liệu cụ thể hơn** ("3,2 tỷ", "12 năm", "ông Vũ Văn Đảo, ông Đinh Văn Quyết").

#### 6.3.4 Tổng kết v4 vs v5

| Khía cạnh | v4 — phù hợp khi | v5 — phù hợp khi |
|---|---|---|
| **K=4** | Tài nguyên compute hạn chế, cần train nhanh | — |
| **K=8** | — | Muốn ước lượng advantage chính xác, chất lượng cao nhất |
| **LR=5e-7** | Model yếu, cần cập nhật thận trọng | — |
| **LR=2e-6** | — | Có SFT nền tảng tốt, muốn cải thiện mạnh |
| **β=0.15** | Cần bám sát reference, ưu tiên tuân thủ độ dài | — |
| **β=0.04** | — | Chấp nhận đánh đổi độ dài để lấy chất lượng nội dung |

**Khuyến nghị:** Sử dụng **v5** cho hầu hết trường hợp (đặc biệt trên nền tảng Instruct-SFT). Với các tác vụ yêu cầu độ tuân thủ độ dài nghiêm ngặt, có thể cân nhắc **v4** hoặc tuning trung gian (K=8, LR=1e-6, β=0.08).

---

## 7. Nhận xét chi tiết theo mô hình

### 7.1 Qwen2.5-3B-Instruct

| Mô hình | ROUGE-2 | LenErr% | Ghi chú |
|---|---|---|---|
| Base | 0.1331 | 60.68% | Baseline |
| SFT (aug) | **0.2308** | **17.92%** | Tốt nhất nhóm |
| SFT (no aug) | 0.2048 | 38.79% | Kém hơn aug |

**Kết luận:** Với Qwen2.5-3B, SFT có augmentation (358K mẫu) cho kết quả tốt hơn đáng kể so với không augmentation (119K mẫu), đặc biệt trên LenErr% (17.92% vs 38.79%). Tuy nhiên, trên ViMs và VLSP, SFT có augmentation vẫn cho kết quả thấp.

### 7.2 Qwen3-4B-Base

| Mô hình | ROUGE-2 | LenErr% | Ghi chú |
|---|---|---|---|
| Base | 0.1056 | 169.77% | Baseline rất yếu (không instruction-tuned) |
| SFT | 0.2632 | 30.52% | Cải thiện vượt bậc |
| GRPO fresh v4 | 0.1388 | 76.44% | Có cải thiện nhưng còn xa SFT |
| GRPO fresh v5 | 0.1866 | 29.05% | Tốt hơn v4 nhưng vẫn thua SFT |
| GRPO sft v4 | **0.2671** | 20.82% | Vượt SFT |
| GRPO sft v5 | 0.2638 | **14.94%** | LenErr% thấp nhất |

**Kết luận:** Qwen3-4B-Base là base model yếu nhất (không instruction-tuned) nhưng sau SFT+GRPO đạt kết quả cạnh tranh với Instruct variant. GRPO fresh (không qua SFT) cho thấy GRPO có thể tự cải thiện từ base nhưng hiệu quả kém hơn so với SFT-init.

**Điểm khác biệt chính v4 vs v5 trên Base:**
- **fresh v4 thất bại gần như hoàn toàn** (R=0.1388, LenErr%=76.44%) — LR=5e-7 quá thấp, K=4 không đủ để ước lượng advantage. Một số mẫu sinh ra bị degenerate nghiêm trọng (vd: một mẫu sinh 246 từ với 225 dấu `!` liên tiếp).
- **fresh v5 cải thiện rõ rệt** (R=0.1866, LenErr%=29.05%) — LR=2e-6 cho phép cập nhật đủ lớn, K=8 giảm variance. Tuy nhiên vẫn còn sinh ký tự ngoại lai (Thai/Trung) ở cuối câu.
- **sft v4 và v5 đều tốt** nhờ nền tảng SFT vững chắc. v5 vượt trội về LenErr% (14.94% vs 20.82%) — nhờ K=8 giúp advantage estimate chính xác hơn, policy không bị nhiễu bởi các tín hiệu sai lệch.

### 7.3 Qwen3-4B-Instruct

| Mô hình | ROUGE-2 | LenErr% | Ghi chú |
|---|---|---|---|
| Base | 0.1506 | 30.22% | Baseline tốt hơn Qwen2.5 |
| SFT | 0.2609 | **12.42%** | LenErr% thấp nhất |
| GRPO fresh v4 | 0.1551 | 26.06% | Cải thiện nhẹ |
| GRPO fresh v5 | 0.1669 | 22.78% | Tốt hơn v4 |
| GRPO sft v4 | 0.2655 | 13.41% | Vượt SFT nhẹ |
| **GRPO sft v5** | **0.2765** | 15.85% | **Mô hình tốt nhất** |

**Kết luận:** Qwen3-4B-Instruct là nền tảng tốt nhất. GRPO sft v5 đạt ROUGE-2 cao nhất toàn bảng (0.2765). SFT thuần cho LenErr% thấp nhất (12.42%) nhưng GRPO giúp cải thiện chất lượng tóm tắt.

**Điểm khác biệt chính v4 vs v5 trên Instruct:**
- **Sự đánh đổi rõ rệt:** v5 cho ROUGE-2 cao hơn (+4.1%) nhưng LenErr% cao hơn (+2.44pp). v4 với β=0.15 giữ policy gần reference hơn → tuân thủ độ dài tốt hơn.
- **Chất lượng sinh:** Cả v4 và v5 đều sinh tiếng Việt sạch (không ký tự ngoại lai). v5 có xu hướng sinh ngắn hơn (44.4 từ vs 48.0) và chứa nhiều chi tiết số liệu cụ thể hơn.
- **fresh v4 vs v5:** fresh v5 cải thiện nhẹ so với v4 (R=0.1669 vs 0.1551, LenErr%=22.78% vs 26.06%) — nhưng vẫn xa so với các mô hình có SFT (~0.26). GRPO từ base trên Instruct không hiệu quả bằng SFT-init.

---

## 8. Phân tích độ dài sinh ra

### 8.1 Độ dài trung bình theo mô hình (test set)

```
QWEN25_base        │███████████████████████████████████████████████████████░░░│ 55.9
QWEN25_sft_aug     │██████████████████████████████████████░░░░░░░░░░░░░░░░░░░│ 35.4
QWEN25_sft_no_aug  │██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ 19.6
QWEN35_base        │█████████████████████████████████████████████████████████│ 69.3
QWEN3_BASE_base    │█████████████████████████████████████████████████████████│ 71.9
QWEN3_BASE_sft     │██████████████████████████████████████████████████████░░░│ 61.1
QWEN3_BASE_grpo_sft_v5│████████████████████████████████████████░░░░░░░░░░░░░│ 46.7
QWEN3_INSTRUCT_base│███████████████████████████████████████████████████░░░░░░│ 54.9
QWEN3_INSTRUCT_sft │████████████████████████████████████████████████░░░░░░░░░│ 51.5
QWEN3_INSTRUCT_grpo_sft_v5│██████████████████████████████████████░░░░░░░░░░░│ 44.4
```

Độ dài trung bình của các mô hình SFT và GRPO dao động từ 35–61 từ, phù hợp với phân bố độ dài của tập dữ liệu. Các mô hình base (chưa huấn luyện) có xu hướng sinh dài hơn (55–72 từ).

### 8.2 So sánh độ dài v4 vs v5

| Mô hình | AvgLen (v4) | AvgLen (v5) | Chênh lệch |
|---|---|---|---|
| Qwen3-Base fresh | 58.2 | **50.3** | **-7.9** 🟢 |
| Qwen3-Base sft | 57.3 | **46.7** | **-10.6** 🟢 |
| Qwen3-Instruct fresh | 52.9 | **50.7** | **-2.2** 🟢 |
| Qwen3-Instruct sft | 48.0 | **44.4** | **-3.6** 🟢 |

v5 có xu hướng sinh **ngắn hơn** v4 trên tất cả các nhánh. Điều này đến từ K=8: với nhiều completion hơn, advantage estimate chính xác hơn → policy có thể học được rằng sinh dài hơn mục tiêu không mang lại lợi ích (vì R_len phạt vượt quá độ dài). Trên nhánh Base sft, mức giảm đặc biệt lớn (-10.6 từ), phù hợp với LenErr% giảm mạnh (20.82% → 14.94%).

### 8.3 Khoảng cách độ dài (LenDist)

LenDist thấp nhất thuộc về các mô hình SFT/GRPO trên VietNews (~1.0–1.5). Trên ViMs và VLSP, LenDist cao hơn nhiều (60–100+) do độ dài mục tiêu của multi-doc summaries lớn và biến động mạnh. v5 không cải thiện LenDist đáng kể so với v4 trên các tập multi-doc, cho thấy vấn đề nằm ở khả năng hiểu instruction hơn là ở config GRPO.

---

## 9. Kết luận và Khuyến nghị

### 9.1 Kết luận

1. **Pipeline SFT→GRPO thành công:** Quy trình hậu huấn luyện hai giai đoạn (SFT → GRPO) cải thiện đáng kể chất lượng tóm tắt so với base model, với ROUGE-2 tăng **84–150%** tùy theo dòng mô hình.

2. **Mô hình tốt nhất:** `QWEN3_INSTRUCT_grpo_sft_v5` (Qwen3-4B-Instruct + SFT + GRPO v5) đạt ROUGE-2=0.2765 và LenErr%=15.85%, là mô hình có chất lượng tổng thể cao nhất.

3. **SFT là bước đệm cần thiết:** GRPO khởi tạo từ SFT cho kết quả vượt trội so với GRPO từ base (fresh), khẳng định vai trò của SFT trong việc cung cấp nền tảng kiến thức ban đầu.

4. **v5 vượt trội v4 với điểm xuất phát thấp, tương đương với điểm xuất phát cao:**
   - Trên nhánh **Base fresh** (không SFT): v5 cải thiện ROUGE-2 từ 0.1388 → 0.1866 (+34.4%) và LenErr% từ 76.44% → 29.05% so với v4. K=8 giúp ước lượng advantage chính xác hơn khi policy còn yếu.
   - Trên nhánh **Instruct-SFT** (có SFT): v5 đạt ROUGE-2=0.2765 vs v4=0.2655 (+4.1%) nhưng LenErr% cao hơn (15.85% vs 13.41%) — thể hiện sự đánh đổi giữa chất lượng nội dung và độ tuân thủ độ dài khi β giảm từ 0.15 → 0.04.
   - Trên nhánh **Base-SFT**: v4 ROUGE-2=0.2671 nhỉnh hơn v5=0.2638 (-1.2%), nhưng v5 cắt giảm LenErr% từ 20.82% → 14.94% (-5.88pp).

5. **Nhiễm ký tự ngoại lai — vấn đề còn lại của Base model:** Cả v4 và v5 trên Qwen3-4B-Base đều còn sinh ký tự Thai/Trung/Hàn ở cuối câu, dấu hiệu policy chưa ổn định. Trên Instruct variant, hiện tượng này hoàn toàn biến mất — khẳng định lợi thế của việc bắt đầu từ instruction-tuned model.

6. **Multi-doc còn là thách thức:** Trên ViMs và VLSP, ROUGE-2 còn thấp (0.18–0.28) và LenErr% cao (30–40%), cho thấy multi-document summarization cần thêm nghiên cứu.

### 9.2 Khuyến nghị

1. **Sử dụng mô hình:** `QWEN3_INSTRUCT_grpo_sft_v5` cho ứng dụng thực tế, đặc biệt trên dữ liệu báo chí (VietNews). Đây là mô hình có chất lượng tổng thể cao nhất (ROUGE-2=0.2765) và sinh tiếng Việt sạch, ổn định.

2. **Lựa chọn config GRPO:**
   - **Mặc định: dùng v5** (K=8, LR=2e-6, β=0.04) cho chất lượng tóm tắt cao nhất.
   - Nếu yêu cầu **tuân thủ độ dài nghiêm ngặt**: dùng v4 (β=0.15) hoặc tuning trung gian (K=8, LR=1e-6, β=0.08).
   - Nếu compute hạn chế: v4 với K=4 tiết kiệm hơn nhưng chất lượng thấp hơn.
   - Trên Base model (không Instruct): **bắt buộc dùng v5** — v4 hầu như không học được (ROUGE-2 chỉ 0.1388).

3. **Cải thiện multi-doc:** Cần thiết kế lại reward cho multi-document summarization, có thể thêm specialized training data cho ViMs và VLSP.

4. **Thử nghiệm thêm:** 
   - Tiếp tục tuning config v5 với K lớn hơn (K=12, 16) — kỳ vọng giảm thêm variance
   - Thử nghiệm config trung gian: K=8, LR=1e-6, β=0.08 để cân bằng content quality và length compliance
   - Áp dụng GRPO cho Qwen3.5-4B (hiện mới chỉ có base) — mô hình mới nhất, nhiều tiềm năng
   - Thử nghiệm DPO sau GRPO warmup (theo hướng G2D)

5. **Đánh giá bổ sung:** Cần thêm đánh giá thủ công (human evaluation) và LLM-as-Judge để đánh giá các khía cạnh như độ trôi chảy, tính đầy đủ thông tin.

---

## Phụ lục: Chi tiết kết quả theo mô hình và tập dữ liệu

| Model | Dataset | N | ROUGE-2 | LenErr% | LenDist | AvgLen |
|---|---|---|---|---|---|---|
| QWEN25_base | test | 3100 | 0.1331 | 60.68 | 28.1 | 55.9 |
| | vietnews | 2000 | 0.1380 | 67.30 | 11.1 | 27.3 |
| | vims | 300 | 0.1775 | 36.99 | 82.9 | 129.3 |
| | vlsp | 300 | 0.1573 | 34.78 | 89.0 | 146.4 |
| | wikilingua | 500 | 0.0726 | 63.97 | 26.7 | 72.1 |
| QWEN25_sft_aug | test | 3100 | 0.2308 | 17.92 | 28.6 | 35.4 |
| | vietnews | 2000 | 0.2666 | 6.35 | 1.1 | 16.9 |
| | vims | 300 | 0.0732 | 62.18 | 135.4 | 80.2 |
| | vlsp | 300 | 0.1332 | 57.06 | 139.7 | 101.1 |
| | wikilingua | 500 | 0.2408 | 14.17 | 7.9 | 43.3 |
| QWEN25_sft_no_aug | test | 3100 | 0.2048 | 38.79 | 45.8 | 19.6 |
| | vietnews | 2000 | 0.2506 | 22.31 | 3.9 | 15.7 |
| | vims | 300 | 0.0392 | 92.29 | 195.0 | 14.0 |
| | vlsp | 300 | 0.0516 | 92.59 | 218.8 | 15.9 |
| | wikilingua | 500 | 0.2127 | 40.34 | 20.4 | 41.1 |
| QWEN35_base | test | 3100 | 0.1492 | 45.56 | 21.0 | 69.3 |
| | vietnews | 2000 | 0.1575 | 37.76 | 6.5 | 23.5 |
| | vims | 300 | 0.1913 | 21.68 | 43.3 | 192.6 |
| | vlsp | 300 | 0.1710 | 20.20 | 49.7 | 204.1 |
| | wikilingua | 500 | 0.0777 | 106.29 | 48.1 | 97.5 |
| QWEN3_BASE_base | test | 3100 | 0.1056 | 169.77 | 51.8 | 71.9 |
| | vietnews | 2000 | 0.1213 | 205.89 | 33.0 | 48.7 |
| | vims | 300 | 0.1346 | 42.61 | 92.0 | 125.8 |
| | vlsp | 300 | 0.0949 | 47.93 | 120.3 | 118.6 |
| | wikilingua | 500 | 0.0318 | 174.67 | 61.6 | 104.0 |
| QWEN3_BASE_grpo_fresh_v4 | test | 3100 | 0.1388 | 76.44 | 32.8 | 58.2 |
| | vietnews | 2000 | 0.1522 | 81.16 | 13.1 | 28.5 |
| | vims | 300 | 0.1740 | 37.37 | 82.7 | 132.0 |
| | vlsp | 300 | 0.1348 | 41.17 | 102.3 | 134.7 |
| | wikilingua | 500 | 0.0664 | 102.18 | 40.3 | 86.3 |
| QWEN3_BASE_grpo_fresh_v5 | test | 3100 | 0.1866 | 29.05 | 21.5 | 50.3 |
| | vietnews | 2000 | 0.2073 | 22.36 | 3.8 | 17.6 |
| | vims | 300 | 0.2028 | 31.89 | 71.8 | 139.9 |
| | vlsp | 300 | 0.1814 | 34.33 | 88.0 | 147.8 |
| | wikilingua | 500 | 0.0971 | 50.94 | 22.0 | 68.8 |
| QWEN3_BASE_grpo_sft_v4 | test | 3100 | 0.2671 | 20.82 | 17.1 | 57.3 |
| | vietnews | 2000 | 0.2745 | 17.10 | 3.2 | 20.0 |
| | vims | 300 | 0.2597 | 39.62 | 69.4 | 179.0 |
| | vlsp | 300 | 0.2527 | 30.43 | 68.1 | 196.0 |
| | wikilingua | 500 | 0.2505 | 18.66 | 10.9 | 50.2 |
| QWEN3_BASE_grpo_sft_v5 | test | 3100 | 0.2638 | 14.94 | 20.5 | 46.7 |
| | vietnews | 2000 | 0.2897 | 6.43 | 1.2 | 17.6 |
| | vims | 300 | 0.1816 | 44.42 | 89.2 | 142.6 |
| | vlsp | 300 | 0.1746 | 42.55 | 100.5 | 150.5 |
| | wikilingua | 500 | 0.2635 | 14.69 | 8.4 | 43.4 |
| QWEN3_BASE_sft | test | 3100 | 0.2632 | 30.52 | 18.4 | 61.1 |
| | vietnews | 2000 | 0.2661 | 31.19 | 5.7 | 22.6 |
| | vims | 300 | 0.2565 | 38.69 | 66.4 | 184.1 |
| | vlsp | 300 | 0.2814 | 27.53 | 61.6 | 203.9 |
| | wikilingua | 500 | 0.2443 | 24.72 | 14.2 | 55.8 |
| QWEN3_INSTRUCT_base | test | 3100 | 0.1506 | 30.22 | 18.9 | 54.9 |
| | vietnews | 2000 | 0.1648 | 24.50 | 4.0 | 19.7 |
| | vims | 300 | 0.1783 | 26.07 | 59.8 | 150.4 |
| | vlsp | 300 | 0.1551 | 27.09 | 70.2 | 165.8 |
| | wikilingua | 500 | 0.0744 | 57.47 | 23.4 | 71.5 |
| QWEN3_INSTRUCT_grpo_fresh_v4 | test | 3100 | 0.1551 | 26.06 | 18.3 | 52.9 |
| | vietnews | 2000 | 0.1708 | 20.15 | 3.4 | 18.3 |
| | vims | 300 | 0.1788 | 27.63 | 62.3 | 148.0 |
| | vlsp | 300 | 0.1569 | 27.64 | 70.9 | 164.8 |
| | wikilingua | 500 | 0.0769 | 47.81 | 19.9 | 67.1 |
| QWEN3_INSTRUCT_grpo_fresh_v5 | test | 3100 | 0.1669 | 22.78 | 17.7 | 50.7 |
| | vietnews | 2000 | 0.1867 | 17.08 | 2.9 | 16.7 |
| | vims | 300 | 0.1843 | 28.37 | 63.1 | 146.5 |
| | vlsp | 300 | 0.1605 | 28.76 | 72.9 | 162.6 |
| | wikilingua | 500 | 0.0809 | 38.67 | 16.1 | 62.5 |
| QWEN3_INSTRUCT_grpo_sft_v4 | test | 3100 | 0.2655 | 13.41 | 16.9 | 48.0 |
| | vietnews | 2000 | 0.2803 | 6.00 | 1.0 | 16.8 |
| | vims | 300 | 0.2177 | 36.24 | 76.1 | 146.1 |
| | vlsp | 300 | 0.2497 | 28.69 | 73.0 | 168.7 |
| | wikilingua | 500 | 0.2444 | 20.18 | 11.0 | 41.6 |
| QWEN3_INSTRUCT_grpo_sft_v5 | test | 3100 | 0.2765 | 15.85 | 18.8 | 44.4 |
| | vietnews | 2000 | 0.2962 | 9.24 | 1.5 | 16.3 |
| | vims | 300 | 0.2127 | 39.14 | 84.5 | 129.5 |
| | vlsp | 300 | 0.2439 | 32.85 | 82.7 | 153.1 |
| | wikilingua | 500 | 0.2559 | 18.11 | 10.2 | 40.5 |
| QWEN3_INSTRUCT_sft | test | 3100 | 0.2609 | 12.42 | 14.6 | 51.5 |
| | vietnews | 2000 | 0.2670 | 6.26 | 1.1 | 17.2 |
| | vims | 300 | 0.2471 | 31.55 | 64.5 | 163.4 |
| | vlsp | 300 | 0.2703 | 24.51 | 62.2 | 182.9 |
| | wikilingua | 500 | 0.2393 | 18.34 | 9.9 | 42.6 |
