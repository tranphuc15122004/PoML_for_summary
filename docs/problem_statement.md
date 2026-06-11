**ĐỀ CƯƠNG CHI TIẾT ĐỀ TÀI NGHIÊN CỨU & PHÁT TRIỂN**

**Tên đề tài:** Hậu huấn luyện (Post-training) cho Mô hình Tóm tắt Văn bản Tiếng Việt theo Yêu cầu Đa dạng

**Đơn vị chủ trì:** Viettel AI

**Lĩnh vực ứng dụng:** Trợ lý ảo (Virtual Assistant), Social 360 và các giải pháp xử lý ngôn ngữ tự nhiên cho doanh nghiệp.

1. **TỔNG QUAN VÀ BỐI CẢNH**

Viettel AI đang đẩy mạnh chiến lược xây dựng và làm chủ các mô hình ngôn ngữ lớn (LLM) phục vụ cho tác vụ tổng hợp và tóm tắt văn bản. Các mô hình này cần đảm bảo các tiêu chuẩn khắt khe về tính bảo mật, độ chính xác và khả năng tùy biến cao nhằm đáp ứng nhu cầu thực tế của khách hàng trong nhiều lĩnh vực, đặc biệt là các sản phẩm cốt lõi như Trợ lý ảo và Social 360.

1. **ĐẶT VẤN ĐỀ VÀ MỤC TIÊU**

**2.1. Đặt vấn đề**

Trong các ứng dụng thực tế, yêu cầu của người dùng về chất lượng và hình thức tóm tắt văn bản rất đa dạng và phụ thuộc nhiều vào ngữ cảnh sử dụng. Các mô hình LLM cơ bản (base models) hoặc chỉ được huấn luyện sơ bộ thường gặp khó khăn trong việc tuân thủ chính xác các ràng buộc phức tạp về độ dài, phong cách ngôn ngữ và định dạng đầu ra, đặc biệt là trên dữ liệu tiếng Việt.

**2.2. Mục tiêu đề tài**

Xây dựng và tối ưu hóa quy trình hậu huấn luyện (post-training) để căn chỉnh hành vi (Alignment) của mô hình dựa trên phản hồi, giúp mô hình tuân thủ tuyệt đối các yêu cầu cụ thể của người dùng, bao gồm:

- **Kiểm soát độ dài (Length Control):** Khả năng tóm tắt văn bản theo số lượng câu hoặc số lượng từ được chỉ định chính xác.
- **Kiểm soát phong cách/ngữ cảnh (Persona Control):** Khả năng điều chỉnh giọng văn và văn phong theo các yêu cầu cụ thể: Báo chí, Sinh hoạt, Chính luận, Khoa học, Hành chính.
- **Kiểm soát định dạng đầu ra (Structured Output Control):** Khả năng xuất kết quả dưới các định dạng có cấu trúc chặt chẽ như danh sách (bullet points) hoặc dữ liệu có định dạng (JSON) mà không bị sai cú pháp hoặc "ảo giác" (hallucination).
1. **PHẠM VI NGHIÊN CỨU**

Để giải quyết bài toán một cách hệ thống, đề tài được chia thành các hướng nghiên cứu trọng tâm:

- **Hướng nghiên cứu 3.1:** Tập trung vào *Length Control* và *Persona Control*, giải quyết bài toán hiểu và tuân thủ các ràng buộc về ngữ nghĩa và độ dài.
- **Hướng nghiên cứu 3.2:** Tập trung vào *Structured Output Control*, giải quyết bài toán đảm bảo tính chính xác về mặt cú pháp và cấu trúc của đầu ra (đặc biệt là JSON).
- **Có thể đề xuất thêm hướng nghiên cứu cho đề tài này**
1. **THÁCH THỨC VÀ GIẢI PHÁP ĐỀ XUẤT**

Quá trình nghiên cứu và phát triển đối mặt với 4 thách thức chính, kèm theo các giải pháp kỹ thuật đề xuất:

1. **Thách thức về Dữ liệu:** Thiếu hụt dữ liệu chất lượng cao cho tiếng Việt, đặc biệt là dữ liệu có kèm lập luận (reasoning data) phục vụ cho tác vụ tóm tắt có ràng buộc.
2. **Thách thức về Mô hình:** Các mô hình LLM kích thước nhỏ (Small LLMs) thường có khả năng tuân thủ chỉ dẫn (instruction-following) kém hơn so với các mô hình lớn.
3. **Thách thức về Thước đo (Metrics):** Các thang đo tự động truyền thống như ROUGE hay BARTScore có hạn chế trong việc đánh giá độ tuân thủ định dạng (JSON), độ dài chính xác và phong cách văn bản.
4. **Thách thức về Tài nguyên:** Thời gian huấn luyện Reinforcement Learning (RL) kéo dài và rủi ro quên thông tin nền tảng (Catastrophic Forgetting).
5. **QUY TRÌNH THỰC HIỆN CHI TIẾT**

Đề tài được triển khai theo 5 giai đoạn tuần tự và logic:

- **Giai đoạn 1: Chuẩn bị dữ liệu & Thiết lập Baseline**
    - Thu thập, làm sạch và phân loại dữ liệu văn bản tiếng Việt đa lĩnh vực.
    - Tạo cặp dữ liệu Instruction-Response với các ràng buộc cụ thể (độ dài, phong cách, format).
    - Thiết lập mô hình Baseline và đo lường các chỉ số hiệu năng ban đầu để làm mốc so sánh.
- **Giai đoạn 2: Supervised Fine-Tuning (SFT)**
    - Thực hiện fine-tuning mô hình (sử dụng LoRA/QLoRA).
    - Mục tiêu: Giúp mô hình nắm bắt hành vi cơ bản, hiểu được instruction và tuân thủ định dạng đầu ra (JSON, bullet) ở mức độ chấp nhận được.
- **Giai đoạn 3: Căn chỉnh hành vi (Alignment Training)**
    - Xây dựng bộ dữ liệu ưu tiên (Preference Data): Cặp (Chosen, Rejected) dựa trên mức độ tuân thủ yêu cầu của người dùng.
    - Huấn luyện mô hình bằng các thuật toán RLHF, DPO hoặc GRPO.
    - Mục tiêu: Tối ưu hóa phản hồi của mô hình dựa trên feedback, nâng cao độ chính xác, độ ổn định và khả năng tuân thủ các ràng buộc phức tạp.
- **Giai đoạn 4: Đánh giá chất lượng mô hình sau Post-training**
    - Đánh giá tự động: Sử dụng ROUGE, BARTScore, và LLM-as-a-Judge (đánh giá độ tuân thủ format, độ dài, phong cách).
    - Đánh giá thủ công: Lấy mẫu kết quả để chuyên gia ngôn ngữ và nghiệp vụ chấm điểm theo rubrics.
    - Kiểm tra hiện tượng Catastrophic Forgetting trên các tác vụ tổng quát.
- **Giai đoạn 5: Xây dựng chương trình Demo**
    - Phát triển giao diện người dùng (UI) cho phép nhập văn bản gốc và cấu hình các tham số yêu cầu (độ dài, phong cách, định dạng đầu ra).
    - Tích hợp API của mô hình đã huấn luyện để hiển thị kết quả tóm tắt theo thời gian thực.
1. **YÊU CẦU ĐẦU RA (DELIVERABLES)**
2. **Mã nguồn & Chương trình Demo:**
    - Source code đầy đủ, được ghi chú rõ ràng cho quy trình tiền xử lý dữ liệu, huấn luyện (SFT, DPO/GRPO) và inference.
    - Ứng dụng Demo (Web App) minh họa trực quan khả năng tóm tắt theo yêu cầu đa dạng của mô hình.
3. **Báo cáo kỹ thuật:**
    - Tài liệu mô tả chi tiết phương pháp thực hiện, kiến trúc dữ liệu và quy trình huấn luyện.
    - Bảng kết quả thử nghiệm, phân tích và so sánh hiệu năng giữa mô hình Baseline và mô hình sau Post-training trên các metrics đã đề ra.
4. **Tài liệu thuyết trình:**
    - Bộ slide tổng hợp toàn bộ nội dung dự án (Bối cảnh, Mục tiêu, Thách thức, Giải pháp, Kết quả và Demo) phục vụ cho việc báo cáo và nghiệm thu đề tài.
5. **TÀI LIỆU THAM KHẢO**
6. Kumar, K., Ashraf, T., Thawakar, O., Anwer, R. M., Cholakkal, H., Shah, M., ... & Khan, S. (2025). *Llm post-training: A deep dive into reasoning large language models*. arXiv preprint arXiv:2502.21321.
7. Tie, G., Zhao, Z., Song, D., Wei, F., Zhou, R., Dai, Y., ... & Gao, J. (2025). *A survey on post-training of large language models*. arXiv preprint arXiv:2503.06072.

*Lưu ý: Tài liệu này đóng vai trò là bản mô tả tổng thể và chi tiết của đề tài. Các thông số kỹ thuật cụ thể (tên mô hình base, kích thước tham số, quy mô tập dữ liệu, hyperparameters) sẽ được bổ sung và cập nhật trong Báo cáo kỹ thuật chi tiết ở giai đoạn thực thi.*