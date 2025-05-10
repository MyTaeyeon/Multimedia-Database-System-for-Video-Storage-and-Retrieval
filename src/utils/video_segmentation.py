import cv2
import numpy as np
import os

# --- Ngưỡng ---
# Ngưỡng cho khác biệt histogram màu để phát hiện chuyển cảnh (cut)
HIST_DIFF_THRESHOLD_CUT = 0.7  # Thử nghiệm với giá trị này, có thể cần điều chỉnh
# Ngưỡng cho độ lớn trung bình của vector optical flow
OPTICAL_FLOW_MAGNITUDE_THRESHOLD = 1.0 # Pixel di chuyển trung bình
# Ngưỡng cho phương sai của góc optical flow (để phát hiện chuyển động đồng nhất như lia máy)
OPTICAL_FLOW_ANGLE_VARIANCE_THRESHOLD = 500 # Độ phân tán của hướng di chuyển
# Ngưỡng cho sự thay đổi độ sáng trung bình (kênh V)
BRIGHTNESS_CHANGE_THRESHOLD = 40 # Trên thang 0-255
# Số khung hình tối thiểu cho một shot
MIN_SHOT_DURATION_FRAMES = 15 # Khoảng 0.5 giây nếu 30fps

def calculate_color_histogram(frame, bins_h=30, bins_s=32, bins_v=32):
    """Tính histogram màu HSV cho một khung hình."""
    if frame is None:
        return None
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Tính histogram cho kênh H, S, V và chuẩn hóa
    hist_h = cv2.calcHist([hsv_frame], [0], None, [bins_h], [0, 180])
    hist_s = cv2.calcHist([hsv_frame], [1], None, [bins_s], [0, 256])
    hist_v = cv2.calcHist([hsv_frame], [2], None, [bins_v], [0, 256])

    cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_s, hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_v, hist_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Nối các histogram lại
    hist = np.concatenate((hist_h, hist_s, hist_v)).flatten()
    return hist

def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    """So sánh hai histogram. Giá trị càng cao (cho CORREL) hoặc càng thấp (cho CHISQR) thì càng giống nhau."""
    if hist1 is None or hist2 is None:
        return 0
    # Sử dụng cv2.HISTCMP_BHATTACHARYYA hoặc cv2.HISTCMP_CHISQR cho khoảng cách
    # cv2.HISTCMP_CORREL cho độ tương quan (càng gần 1 càng tốt)
    # Chúng ta muốn phát hiện sự *khác biệt*, nên 1 - correlation
    # Hoặc nếu dùng Chi-squared, giá trị lớn hơn nghĩa là khác biệt hơn
    similarity = cv2.compareHist(hist1, hist2, method)
    if method == cv2.HISTCMP_CORREL or method == cv2.HISTCMP_INTERSECT:
        return 1 - similarity # Để giá trị lớn hơn nghĩa là khác biệt hơn
    return similarity # Cho Chi-squared, Bhattacharyya

def calculate_optical_flow_features(prev_gray, current_gray):
    """
    Tính toán các đặc trưng từ luồng quang học.
    Trả về: độ lớn trung bình của luồng, phương sai của góc luồng.
    """
    if prev_gray is None or current_gray is None:
        return 0, float('inf')

    # Tính toán dense optical flow bằng thuật toán Farneback
    # params: prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
    # pyr_scale: tỷ lệ thu nhỏ ảnh cho mỗi tầng kim tự tháp (0.5 là cổ điển)
    # levels: số tầng kim tự tháp
    # winsize: kích thước cửa sổ trung bình (lớn hơn thì robust với nhiễu, nhưng làm mờ chuyển động)
    # iterations: số lần lặp lại thuật toán trên mỗi tầng kim tự tháp
    # poly_n: kích thước vùng lân cận để xấp xỉ đa thức (5 hoặc 7)
    # poly_sigma: độ lệch chuẩn của Gaussian dùng để làm mịn (1.1 cho poly_n=5, 1.5 cho poly_n=7)
    # flags: có thể là 0 hoặc cv2.OPTFLOW_USE_INITIAL_FLOW
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2,
                                        flags=0)
    if flow is None:
        return 0, float('inf')

    # Tách thành phần x, y của vector luồng
    fx, fy = flow[..., 0], flow[..., 1]

    # Tính độ lớn (magnitude) và góc (angle) của vector luồng
    magnitude, angle_rad = cv2.cartToPolar(fx, fy)
    angle_deg = angle_rad * 180 / np.pi / 2 # Chuyển sang độ (0-360 rồi chia 2 để khớp với hue 0-180)

    mean_magnitude = np.mean(magnitude)
    angle_variance = np.var(angle_deg) # Phương sai của góc

    return mean_magnitude, angle_variance

def get_brightness_difference(frame1, frame2):
    """Tính khác biệt độ sáng trung bình (kênh V)."""
    if frame1 is None or frame2 is None:
        return 0
    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    mean_v1 = np.mean(hsv1[:, :, 2])
    mean_v2 = np.mean(hsv2[:, :, 2])
    return abs(mean_v1 - mean_v2)

def preprocess_frame(frame, target_width=320): # Hàm resize
    if frame is None: return None
    h, w = frame.shape[:2]
    if w > target_width:
        scale_ratio = target_width / w
        new_height = int(h * scale_ratio)
        return cv2.resize(frame, (target_width, new_height), interpolation=cv2.INTER_AREA)
    return frame

def segment_video_into_shots(video_path,keyframe_save_dir):
    """
    Phân đoạn video thành các shot và chọn keyframe cho mỗi shot.
    Trả về một danh sách các keyframe (dưới dạng mảng NumPy).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: # Một số video có thể không trả về fps đúng, gán giá trị mặc định
        print(f"Cảnh báo: Không thể lấy FPS từ video {video_path}. Sử dụng FPS mặc định = 25.0")
        fps = 25.0

    # keyframes = []
    processed_shots_info = []
    # shot_boundaries = [0] # Mảng lưu chỉ số khung hình bắt đầu của mỗi shot

    ret, prev_frame_orig = cap.read()
    if not ret:
        print(f"Lỗi: Không thể đọc frame đầu tiên từ video {video_path}")
        cap.release()
        return [],fps

    prev_frame = preprocess_frame(prev_frame_orig.copy())
    prev_hist = calculate_color_histogram(prev_frame)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_brightness = np.mean(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)[:,:,2])

    frame_count = 0
    # last_shot_frame_index = 0
    current_shot_start_frame_idx = 0
    current_shot_frames_orig = [prev_frame_orig.copy()] # Lưu trữ các frame của shot hiện tại để chọn keyframe

    shot_counter = 0

    while True:
        ret, current_frame_orig = cap.read()
        if not ret:
            break # Kết thúc video

        frame_count += 1
        current_frame= preprocess_frame(current_frame_orig.copy())
        
        if current_frame is None:
            prev_frame_orig = current_frame_orig.copy()
            current_shot_frames_orig.append(prev_frame_orig.copy())
            continue
        
        current_hist = calculate_color_histogram(current_frame)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_brightness = np.mean(cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)[:,:,2])

        hist_diff = compare_histograms(prev_hist, current_hist, method=cv2.HISTCMP_BHATTACHARYYA) # Bhattacharyya: giá trị càng nhỏ càng giống
        # HISTCMP_CORREL thì 1.0 là giống hệt, 0.0 là khác biệt. Ta muốn (1 - corr)
        # Nếu dùng HISTCMP_CHISQR hoặc HISTCMP_BHATTACHARYYA, giá trị lớn hơn nghĩa là khác biệt hơn.
        # HIST_DIFF_THRESHOLD_CUT là ngưỡng cho sự khác biệt (ví dụ: 0.7 cho Bhattacharyya là khá khác nhau)

        # Tính toán optical flow
        flow_magnitude, flow_angle_variance = calculate_optical_flow_features(prev_gray, current_gray)

        # Kiểm tra thay đổi ánh sáng
        brightness_diff = abs(current_brightness - prev_brightness)

        is_potential_cut = hist_diff > HIST_DIFF_THRESHOLD_CUT

        # Phân biệt chuyển cảnh thực sự với lia máy/thay đổi góc và thay đổi ánh sáng
        is_camera_motion = (flow_magnitude > OPTICAL_FLOW_MAGNITUDE_THRESHOLD and
                            flow_angle_variance < OPTICAL_FLOW_ANGLE_VARIANCE_THRESHOLD)

        is_lighting_change_dominant = (brightness_diff > BRIGHTNESS_CHANGE_THRESHOLD and
                                     hist_diff < HIST_DIFF_THRESHOLD_CUT * 1.2 and # Histogram không quá khác nếu chỉ là ánh sáng
                                     not is_camera_motion) # Và không phải do camera di chuyển

        is_true_cut = is_potential_cut and not is_camera_motion and not is_lighting_change_dominant

        if is_true_cut:
            # Shot hiện tại kết thúc ở frame_count- 1
            current_shot_end_frame_idx = frame_count -1
            if (current_shot_end_frame_idx - current_shot_start_frame_idx + 1) >= MIN_SHOT_DURATION_FRAMES:
                # Chọn keyframe cho shot vừa kết thúc
                # Đơn giản nhất là chọn frame ở giữa shot
                if current_shot_frames_orig:
                    keyframe_index = len(current_shot_frames_orig) // 2
                    # keyframes.append(current_shot_frames_orig[keyframe_index].copy())
                    selected_keyframe_image = current_shot_frames_orig[keyframe_index].copy()
                    
                    # Lưu keyframe ra file (tùy chọn)
                    video_basename = os.path.splitext(os.path.basename(video_path))[0]
                    keyframe_filename = f"{video_basename}_shot_{shot_counter}_frame_{current_shot_start_frame_idx + keyframe_index}.jpg"
                    keyframe_save_path = os.path.join(keyframe_save_dir, video_basename, keyframe_filename)
                    os.makedirs(os.path.join(keyframe_save_dir, video_basename), exist_ok=True) # Tạo thư mục con cho mỗi video
                    cv2.imwrite(keyframe_save_path, selected_keyframe_image)
                    
                    shot_info = {
                        "segment_id": f"shot_{shot_counter}",
                        "start_frame_index": current_shot_start_frame_idx,
                        "end_frame_index": current_shot_end_frame_idx,
                        "start_time_sec": current_shot_start_frame_idx / fps,
                        "end_time_sec": (current_shot_end_frame_idx + 1) / fps, # End time là đầu frame tiếp theo
                        "keyframe_image_for_feature": selected_keyframe_image, # Ảnh keyframe để trích rút đặc trưng
                        "keyframe_image_path": keyframe_save_path # Đường dẫn nếu muốn lưu
                    }
                    processed_shots_info.append(shot_info)
                    shot_counter += 1
                # shot_boundaries.append(frame_count)
                # last_shot_frame_index = frame_count
                current_shot_start_frame_idx = frame_count # Bắt đầu shot mới
                current_shot_frames_orig = [] # Bắt đầu shot mới
            # Nếu shot quá ngắn, có thể gộp vào shot trước hoặc bỏ qua
            # Ở đây, chúng ta chỉ ghi nhận shot nếu đủ dài

        current_shot_frames_orig.append(current_frame_orig.copy())
        prev_frame_orig = current_frame_orig.copy()
        prev_frame = current_frame.copy()
        prev_hist = current_hist
        prev_gray = current_gray
        prev_brightness = current_brightness

    # Xử lý shot cuối cùng
    current_shot_end_frame_idx = frame_count
    if current_shot_frames_orig and (current_shot_end_frame_idx - current_shot_start_frame_idx + 1) >= MIN_SHOT_DURATION_FRAMES:
        keyframe_index = len(current_shot_frames_orig) // 2
        # keyframes.append(current_shot_frames_orig[keyframe_index].copy())
        selected_keyframe_image = current_shot_frames_orig[keyframe_index].copy()
        
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        keyframe_filename = f"{video_basename}_shot_{shot_counter}_frame_{current_shot_start_frame_idx + keyframe_index}.jpg"
        keyframe_save_path = os.path.join(keyframe_save_dir, video_basename, keyframe_filename)
        os.makedirs(os.path.join(keyframe_save_dir, video_basename), exist_ok=True)
        cv2.imwrite(keyframe_save_path, selected_keyframe_image)
        
        shot_info = {
            "segment_id": f"shot_{shot_counter}",
            "start_frame_index": current_shot_start_frame_idx,
            "end_frame_index": current_shot_end_frame_idx,
            "start_time_sec": current_shot_start_frame_idx / fps,
            "end_time_sec": (current_shot_end_frame_idx + 1) / fps,
            "keyframe_image_for_feature": selected_keyframe_image,
            "keyframe_image_path": keyframe_save_path
        }
        processed_shots_info.append(shot_info)
        # shot_boundaries.append(frame_count) # Điểm kết thúc của shot cuối

    cap.release()
    # print(f"Video {os.path.basename(video_path)}: Tìm thấy {len(keyframes)} keyframes từ {len(shot_boundaries)-1} shots.")
    # return keyframes
    return processed_shots_info,fps

if __name__ == '__main__':
    # --- Thử nghiệm ---
    sample_video_dir = "../Dataset/LandscapeVideos/"
    output_keyframe_dir = "../ExtractedFeatures/SampleKeyframes/"
    os.makedirs(output_keyframe_dir, exist_ok=True)

    if not os.path.exists(sample_video_dir) or not os.listdir(sample_video_dir):
        print(f"Thư mục video mẫu '{sample_video_dir}' không tồn tại hoặc trống.")
        print("Vui lòng tạo thư mục và đặt một video mẫu (ví dụ: 'sample.mp4') vào đó để thử nghiệm.")
    else:
        sample_video_path = None
        for f_name in os.listdir(sample_video_dir):
            if f_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                sample_video_path = os.path.join(sample_video_dir, f_name)
                break

        if sample_video_path:
            print(f"Đang xử lý video mẫu: {sample_video_path}")
            extracted_keyframes = segment_video_into_shots(sample_video_path)

            if extracted_keyframes:
                print(f"Trích xuất được {len(extracted_keyframes)} keyframes.")
                for i, kf in enumerate(extracted_keyframes):
                    cv2.imwrite(os.path.join(output_keyframe_dir, f"{os.path.basename(sample_video_path)}_keyframe_{i}.jpg"), kf)
                print(f"Đã lưu keyframes vào thư mục: {output_keyframe_dir}")
            else:
                print("Không trích xuất được keyframe nào.")
        else:
            print(f"Không tìm thấy file video nào trong '{sample_video_dir}'.")