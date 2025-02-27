# HỆ CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN

### YÊU CẦU

Xây dựng hệ CSDL lưu trữ và tìm kiếm video.

1. Hãy xây dựng/sưu tầm một bộ dữ liệu gồm ít nhất 100 files video với cùng một chủ đề, có độ dài tối thiểu 10s (SV tùy chọn định dạng file video và nội dung video).
2. Hãy xây dựng một quy trình trích rút đặc trưng để nhận diện nội dung các file video khác nhau từ bộ dữ liệu đã thu thập. Trình bày cụ thể về lý do lựa chọn và giá trị thông tin của các đặc trưng này.
3. Xây dựng hệ thống tìm kiếm video với đầu vào là một file ảnh với nội dung đã có và không có trong các file video dữ liệu, đầu ra là 3 files video chứa nội dung liên quan nhất đến ảnh đầu vào, xếp thứ tự giảm dần về độ tương đồng.

    a. Trình bày sơ đồ khối của hệ thống và quy trình thực hiện yêu cầu của đề bài.

    b. Trình bày quá trình trích rút, lưu trữ và sử dụng các đặc trưng để tìm kiếm video trong hệ thống.
4. Demo hệ thống và đánh giá kết quả đã đạt được.

### Hướng tiếp cận

✅ Xây dựng bộ dữ liệu video theo một chủ đề động vật. (tối thiểu **100 videos** mỗi video dài tối thiểu **10s**) [Link data](https://ptiteduvn-my.sharepoint.com/:f:/g/personal/giangnm_b21cn304_stu_ptit_edu_vn/Enw9hOKXzcdIjt2OMvcoQYkBAchLF_jHwYAWn4_9-GPV_w?e=deATzO)
    
**Dữ liệu train:**         

| **Mô tả** | **Tỉ lệ** |
|-----------|-----------|
| ... | ? %|
| ... | ? %|

**Dữ liệu test:**         

| **Mô tả** | **Tỉ lệ** |
|-----------|-----------|
| ... | ? %|
| ... | ? %|

✅ Trích xuất đặc trưng video bằng Deep Learning và Computer Vision.

✅ Tạo hệ thống tìm kiếm với truy vấn đầu vào là một ảnh.

✅ Demo hệ thống và đánh giá hiệu suất.

### Cài đặt

    conda create -n venv AnimalClfVenv python=3.12
    git clone ...
    python install -r requirements.txt

### Demo

Link sản phẩm: ...