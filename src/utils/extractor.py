import cv2
import os
import compute
import numpy as np

def extract_frames(video_path, output_dir=None, every_n_seconds=1):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Số khung hình/giây
    frame_interval = int(fps * every_n_seconds)

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chỉ lưu khung hình mỗi N giây
        if frame_count % frame_interval == 0:
            frame_name = f"{saved_count:04d}.jpg"
            save_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(save_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Đã lưu {saved_count} khung hình vào thư mục: {output_dir}")

def segment_video_by_histogram(video_path, threshold=0.6):
    cap = cv2.VideoCapture(video_path)
    prev_hist = None
    shot_boundaries = [0]
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_hist = compute.calc_histogram(frame)
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, current_hist, cv2.HISTCMP_CORREL)
            if diff < threshold:
                shot_boundaries.append(frame_count)

        prev_hist = current_hist
        frame_count += 1

    shot_boundaries.append(frame_count)  # Thêm khung cuối cùng

    cap.release()
    return shot_boundaries

def get_keyframe_nearest_avg(cap, start_idx, end_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    frames = []
    for i in range(start_idx, end_idx):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Tính frame trung bình
    avg_frame = np.mean(np.array(frames, dtype=np.float32), axis=0)

    # Tìm frame gần nhất (MSE nhỏ nhất)
    min_mse = float('inf')
    best_idx = start_idx
    for i, frame in enumerate(frames):
        mse = np.mean((frame.astype(np.float32) - avg_frame) ** 2)
        if mse < min_mse:
            min_mse = mse
            best_idx = start_idx + i

    return best_idx

def get_segment_keyframes(video_path, shot_boundaries):
    cap = cv2.VideoCapture(video_path)
    keyframes = []

    for i in range(len(shot_boundaries) - 1):
        start_idx = shot_boundaries[i]
        end_idx = shot_boundaries[i + 1]

        keyframe_idx = get_keyframe_nearest_avg(cap, start_idx, end_idx)
        keyframes.append(keyframe_idx)

    cap.release()
    return keyframes

def extract_hsv_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

    # Chuẩn hóa histogram
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    # Kết hợp các histogram thành một vector đặc trưng
    hsv_features = np.concatenate([hist_h, hist_s, hist_v])
    return hsv_features

def extract_and_save_hsv_features(image_paths, output_file):
    features = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Không thể đọc ảnh: {image_path}")
            continue

        hsv_features = extract_hsv_features(image)
        features.append(hsv_features)

    features = np.array(features)
    np.save(output_file, features)
    print(f"Đã lưu đặc trưng HSV vào file: {output_file}")