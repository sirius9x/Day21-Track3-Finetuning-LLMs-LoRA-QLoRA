# Lab 21 — Evaluation Report

**Học viên**: Đoàn Xuân Thạch — 2A202600950
**Ngày nộp**: 2026-06-25
**Submission option**: A (lightweight)

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval)
- **max_seq_length**: 1024 (p95 = 562, rounded up)
- **GPU**: Tesla T4, 15.6 GB VRAM
- **Training cost**: ~$0.04 (~13 phút @ $0.20/hr)
- **HF Hub link**: N/A

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 1,843,200       | 4.19 min   | 7.21 GB   | 1.5576    | 4.74       |
| 16   | 3,686,400       | 4.56 min   | 6.61 GB   | 1.5160    | 4.55       |
| 64   | 14,745,600      | 4.22 min   | 7.99 GB   | 1.4768    | 4.37       |
| Base | -               | -          | -         | N/A       | N/A        |

## 3. Loss Curve Analysis
![Loss Curve](loss_curve.png)
- Quan sát: Training loss giảm đều đặn từ khoảng 1.61 xuống 1.27 (đối với r=64) qua 69 steps. Loss không có dấu hiệu tăng vọt, cho thấy mô hình đang học tốt. Vì không có đánh giá eval_loss qua từng step nên chưa thể khẳng định chắc chắn có overfitting hay không, tuy nhiên eval perplexity cuối cùng tương đối thấp cho thấy mô hình đã nắm bắt được định dạng instruction tốt trên tập dữ liệu này.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
**Base**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, n...
**Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân ...
**Nhận xét**: Improved (Câu văn của mô hình fine-tuned tự nhiên và sát nghĩa ngôn ngữ tiếng Việt hơn so với bản gốc).

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
**Base**: Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n:\n\n```python\ndef fibonacci(n):\n    if n <= 0:\n        return "N p...
**Fine-tuned (r=16)**: Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau:\n\n```python\ndef fibonacci(n):\n    if n < 0:\n        raise ValueError("Input phải là một số nguyên dương.")\n    elif n == 0:\n   ...
**Nhận xét**: Improved (Mô hình FT kiểm tra lỗi đầu vào (n < 0) chuyên nghiệp hơn bằng cách raise ValueError).

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
**Base**: 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục, màu sắc,...
**Fine-tuned (r=16)**: 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản: UI/UX th...
**Nhận xét**: Improved (Câu trả lời từ mô hình FT đi vào trọng tâm, phân tách thành các ý nhỏ với số thứ tự và gạch đầu dòng rõ ràng).

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
**Base**: LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp cải thiện hi...
**Fine-tuned (r=16)**: LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển để cải thiện hiệu quả và độ ổn định của các mạng neural network trong...
**Nhận xét**: Degraded (Mô hình FT đưa ra khái niệm định nghĩa từ viết tắt của LoRA hoàn toàn sai - một dạng của hallucination).

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
**Base**: Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất của ...
**Fine-tuned (r=16)**: Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp ...
**Nhận xét**: Same (Cả hai mô hình đều hiểu được ngữ cảnh của câu hỏi, nội dung giải thích ban đầu có chất lượng tương đương nhau).

## 5. Conclusion về Rank Trade-off

- **Rank nào cho ROI tốt nhất trên dataset này? Tại sao?**
Rank 16 cho ROI (Return on Investment) tốt nhất. So với r=8, r=16 cải thiện Perplexity đáng kể (giảm từ 4.74 xuống 4.55). Đồng thời, thời gian train (~4.56 min) và VRAM (6.61 GB) của r=16 rất sát với r=8, vô cùng tiết kiệm nhưng chất lượng được tăng thêm. Số lượng tham số cần train (3.6M) vừa đủ để nắm bắt pattern của tập dataset.
- **Khi nào tăng rank không còn cải thiện perplexity (diminishing returns)?**
Khi rank tăng lên 64, số lượng tham số cần train tăng vọt gấp 4 lần (lên ~14.7M) và VRAM yêu cầu tiến sát mức 8GB. Dù Perplexity có giảm tiếp (từ 4.55 xuống 4.37), mức độ cải thiện này không còn tương xứng với tỷ lệ tham số đưa vào huấn luyện (diminishing returns). Việc tăng cao quá trên dataset nhỏ còn mang nguy cơ làm overfitting model.
- **Recommendation: nếu deploy production, bạn chọn rank nào? Tại sao?**
Nếu deploy production cho use-case tương tự, tôi sẽ chọn rank=16. Nó đạt được sự cân bằng tối ưu giữa việc cung cấp đủ tham số để học các đặc trưng ngôn ngữ mới (tự nhiên hơn) nhưng vẫn giữ được dung lượng adapter (weights) ở mức nhỏ gọn. Một adapter nhẹ hỗ trợ việc inference nhanh, ít tốn chi phí host/lưu trữ và đặc biệt hạn chế việc can thiệp/phá vỡ quá nhiều tri thức (general knowledge) vốn có của base model.

## 6. What I Learned
- Sử dụng framework như Unsloth kết hợp QLoRA 4-bit thực sự mang tính cách mạng vì cho phép fine-tune các model tỷ tham số một cách nhẹ nhàng trên những card GPU cấp thấp, miễn phí như T4 mà không gặp lỗi OOM (Out of Memory).
- Đánh giá chất lượng model cần kết hợp cả phương pháp định lượng (Eval Loss, Perplexity) và định tính (đọc thử từng câu sinh ra). Đôi khi Loss giảm, văn phong giống mẫu hơn nhưng thực chất mô hình đang bị "hallucinate" (như trong trường hợp định nghĩa nhầm LoRA ở Example 4).
- Không phải lúc nào số lượng "Trainable Params" lớn (Rank lớn) cũng là giải pháp tốt nhất. Quản lý Rank là yếu tố trade-off quan trọng để cân đối giữa chi phí tính toán, rủi ro overfitting và hiệu quả nâng cấp của mô hình.
