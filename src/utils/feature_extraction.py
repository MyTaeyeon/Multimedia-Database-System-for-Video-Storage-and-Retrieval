import cv2
import numpy as np
from skimage.feature import local_binary_pattern # Cần cài đặt: pip install scikit-image

# --- Tham số cho LBP ---
LBP_RADIUS = 1 # Bán kính của vùng lân cận
LBP_N_POINTS = 8 * LBP_RADIUS # Số điểm lân cận
LBP_METHOD = 'uniform' # 'default', 'ror', 'uniform', 'nri_uniform', 'var'
                      # 'uniform' giúp giảm chiều vector đặc trưng và robust hơn

# --- Tham số cho Đặc trưng Cạnh (Sobel) ---
SOBEL_KERNEL_SIZE = 3 # Kích thước kernel Sobel (thường là 3, 5, 7)
EDGE_HIST_BINS = 8 # Số bin cho histogram hướng cạnh

def extract_hsv_histogram(frame, h_bins=30, s_bins=32, v_bins=32):
    """Trích rút đặc trưng histogram HSV từ một frame."""
    if frame is None:
        return None
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv_frame], [0], None, [h_bins], [0, 180])
    hist_s = cv2.calcHist([hsv_frame], [1], None, [s_bins], [0, 256])
    hist_v = cv2.calcHist([hsv_frame], [2], None, [v_bins], [0, 256])

    cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_s, hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_v, hist_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    histogram = np.concatenate((hist_h, hist_s, hist_v)).flatten()
    return histogram

def extract_lbp_histogram(frame):
    """Trích rút đặc trưng LBP histogram từ một frame."""
    if frame is None:
        return None
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Tính LBP
    lbp = local_binary_pattern(gray_frame, LBP_N_POINTS, LBP_RADIUS, LBP_METHOD)
    # Tính histogram của các mã LBP
    # Số bin cho LBP 'uniform' là n_points + 2
    n_bins_lbp = int(lbp.max() + 1) # Hoặc LBP_N_POINTS + 2 nếu là 'uniform'
    if LBP_METHOD == 'uniform':
        n_bins_lbp = LBP_N_POINTS + 2

    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_bins_lbp + 1),
                             range=(0, n_bins_lbp))
    # Chuẩn hóa histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6) # Thêm epsilon để tránh chia cho 0
    return hist

def extract_edge_features_simplified(frame):
    """
    Trích rút đặc trưng cạnh đơn giản: histogram của hướng gradient.
    """
    if frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Tính gradient theo x và y bằng Sobel
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=SOBEL_KERNEL_SIZE)

    # Tính độ lớn và hướng gradient
    magnitude, angle_rad = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=False)
    angle_deg = np.degrees(angle_rad) % 180 # Hướng không phân biệt (0-180 độ)

    # Tạo histogram của hướng gradient, chỉ xem xét các pixel có độ lớn gradient đáng kể
    # (để giảm nhiễu từ các vùng phẳng)
    significant_edges_mask = magnitude > (0.1 * np.max(magnitude)) # Ngưỡng đơn giản
    significant_angles = angle_deg[significant_edges_mask]

    if significant_angles.size == 0:
        return np.zeros(EDGE_HIST_BINS) # Trả về vector 0 nếu không có cạnh nào đáng kể

    edge_hist, _ = np.histogram(significant_angles,
                                bins=EDGE_HIST_BINS,
                                range=(0, 180))
    # Chuẩn hóa histogram
    edge_hist = edge_hist.astype("float")
    edge_hist /= (edge_hist.sum() + 1e-6)
    return edge_hist

def extract_features_from_keyframe(keyframe):
    """Trích rút tất cả các đặc trưng từ một keyframe và nối chúng lại."""
    if keyframe is None:
        print("Cảnh báo: Keyframe rỗng, không thể trích rút đặc trưng.")
        return None

    hsv_features = extract_hsv_histogram(keyframe)
    lbp_features = extract_lbp_histogram(keyframe)
    edge_features = extract_edge_features_simplified(keyframe)

    if hsv_features is None or lbp_features is None or edge_features is None:
        return None

    # Nối tất cả các vector đặc trưng lại
    combined_features = np.concatenate((hsv_features, lbp_features, edge_features))
    return combined_features


if __name__ == '__main__':
    # --- Thử nghiệm ---
    sample_keyframe_path = "../ExtractedFeatures/SampleKeyframes/" # Giả sử đã có keyframe từ bước trước

    if not os.path.exists(sample_keyframe_path) or not os.listdir(sample_keyframe_path):
        print(f"Thư mục keyframe mẫu '{sample_keyframe_path}' không tồn tại hoặc trống.")
        print("Vui lòng chạy video_segmentation.py trước để tạo keyframes mẫu.")
    else:
        keyframe_files = [f for f in os.listdir(sample_keyframe_path) if f.lower().endswith(('.jpg', '.png'))]
        if keyframe_files:
            test_keyframe_file = os.path.join(sample_keyframe_path, keyframe_files[0])
            print(f"Đang trích rút đặc trưng từ keyframe mẫu: {test_keyframe_file}")
            keyframe_image = cv2.imread(test_keyframe_file)

            if keyframe_image is not None:
                features = extract_features_from_keyframe(keyframe_image)
                if features is not None:
                    print(f"Đặc trưng HSV (một phần): {extract_hsv_histogram(keyframe_image)[:10]}...")
                    print(f"  Chiều vector HSV: {len(extract_hsv_histogram(keyframe_image))}")
                    print(f"Đặc trưng LBP (một phần): {extract_lbp_histogram(keyframe_image)[:10]}...")
                    print(f"  Chiều vector LBP: {len(extract_lbp_histogram(keyframe_image))}")
                    print(f"Đặc trưng Cạnh (một phần): {extract_edge_features_simplified(keyframe_image)}...")
                    print(f"  Chiều vector Cạnh: {len(extract_edge_features_simplified(keyframe_image))}")
                    print(f"Vector đặc trưng tổng hợp (một phần): {features[:20]}...")
                    print(f"Chiều dài vector đặc trưng tổng hợp: {len(features)}")
                else:
                    print("Không thể trích rút đặc trưng từ keyframe mẫu.")
            else:
                print(f"Không thể đọc file keyframe: {test_keyframe_file}")
        else:
            print(f"Không tìm thấy file keyframe nào trong '{sample_keyframe_path}'.")