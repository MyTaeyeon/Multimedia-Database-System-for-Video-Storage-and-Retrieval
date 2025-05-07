# HỆ CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN

[Link thu thập data](https://ptiteduvn-my.sharepoint.com/:f:/g/personal/giangnm_b21cn304_stu_ptit_edu_vn/Enw9hOKXzcdIjt2OMvcoQYkBAchLF_jHwYAWn4_9-GPV_w?e=deATzO)

[Báo cáo](https://ptiteduvn-my.sharepoint.com/:w:/g/personal/giangnm_b21cn304_stu_ptit_edu_vn/ETrfHsIrlf9BrwbXioKucNkBElBn91f8wPy-yJo5uzLUzg?e=xzkIsU)

## YÊU CẦU

Xây dựng hệ CSDL lưu trữ và tìm kiếm video.

1. Hãy xây dựng/sưu tầm một bộ dữ liệu gồm ít nhất 100 files video với cùng một chủ đề, có độ dài tối thiểu 10s (SV tùy chọn định dạng file video và nội dung video).
2. Hãy xây dựng một quy trình trích rút đặc trưng để nhận diện nội dung các file video khác nhau từ bộ dữ liệu đã thu thập. Trình bày cụ thể về lý do lựa chọn và giá trị thông tin của các đặc trưng này.
3. Xây dựng hệ thống tìm kiếm video với đầu vào là một file ảnh với nội dung đã có và không có trong các file video dữ liệu, đầu ra là 3 files video chứa nội dung liên quan nhất đến ảnh đầu vào, xếp thứ tự giảm dần về độ tương đồng.

    a. Trình bày sơ đồ khối của hệ thống và quy trình thực hiện yêu cầu của đề bài.

    b. Trình bày quá trình trích rút, lưu trữ và sử dụng các đặc trưng để tìm kiếm video trong hệ thống.
4. Demo hệ thống và đánh giá kết quả đã đạt được.

## Hướng tiếp cận

✅ Xây dựng bộ dữ liệu video theo một chủ đề động vật. (tối thiểu **100 videos** mỗi video dài tối thiểu **10s**) [Link data](!)

✅ Trích xuất đặc trưng video bằng Deep Learning và Computer Vision.

✅ Tạo hệ thống tìm kiếm với truy vấn đầu vào là một ảnh.

✅ Demo hệ thống và đánh giá hiệu suất.

## Các bước

### Trích xuất đặc trưng
Đầu vào:

Đầu ra:

### Kiến trúc hệ thống
🧱 1. Physical Storage View (Tầng lưu trữ vật lý)

    Đây là nơi lưu trữ dữ liệu đa phương tiện thật sự, gồm:

    📄 Text: lưu các văn bản dạng thuần như mô tả, phụ đề, chú thích,...

    🖼️ Image: lưu ảnh (ảnh JPEG, PNG,...)

    🎥 Video: lưu video (MP4, AVI,...)

    🔊 Audio: lưu âm thanh (MP3, WAV,...)

🧠 2. Conceptual Data View (Tầng dữ liệu khái niệm)

    Đây là tầng trung gian mô hình hóa, truy cập và tổ chức dữ liệu, gồm:

    🧩 Data models:

    Các mô hình biểu diễn nội dung ảnh, video, âm thanh (ví dụ: histogram màu, đặc trưng LBP, waveform,...)

    Có thể là dạng quan hệ, hướng đối tượng, hoặc mô hình đặc thù cho đa phương tiện.

    🔄 Data access:

    Giao diện và cơ chế truy xuất dữ liệu: SQL mở rộng, truy vấn theo nội dung (CBR),...

    ⏱️ Temporal models:

    Mô hình dữ liệu có yếu tố thời gian như video, audio (timeline, đồng bộ, phân cảnh,...)

🌐 3. Distributed View (Tầng phân tán)

    Gồm mạng truyền thông (communication network): là cơ sở để truy vấn và chia sẻ dữ liệu từ nhiều nơi.

    Truy vấn từ người dùng (Query 1, Query 2, Query 3) được gửi qua mạng đến tầng khái niệm → truy cập dữ liệu tương ứng.

👁️ 4. User’s View (Tầng người dùng)

    Biểu diễn giao diện người dùng (W1, W2, W3 là các cửa sổ ứng dụng/mô-đun).

    Mỗi người dùng có thể có nhiều cửa sổ hiển thị nhiều loại dữ liệu khác nhau (ảnh, video, văn bản,...).

    Các truy vấn (Query 1/2/3) được sinh ra từ hành động người dùng trên các cửa sổ này.
## Cài đặt

    conda create -n venv AnimalClfVenv python=3.12
    git clone ...
    python install -r requirements.txt

## Demo

Link demo: ...