import json
import numpy as np
import cv2
from scipy.spatial.distance import cosine,euclidean
# bhattacharyya, euclidean # Các độ đo khoảng cách

# Import các hàm trích rút đặc trưng từ file feature_extraction.py
from utils.feature_extraction import extract_features_from_keyframe, extract_hsv_histogram, extract_lbp_histogram

# --- Cấu hình lưu trữ ---
FEATURE_DB_PATH = "./outputs/extracted_features/features_database.json"

def save_features_db(features_data,path=FEATURE_DB_PATH):
    """Lưu trữ cơ sở dữ liệu đặc trưng vào file JSON."""
    try:
        with open(path, 'w') as f:
            # Chuyển đổi numpy arrays thành list để có thể serialize JSON
            # serializable_data = []
            # for item in features_data:
            #     serializable_item = item.copy() # Tạo bản sao để không thay đổi dữ liệu gốc
            #     serializable_item['features'] = item['features'].tolist()
            #     serializable_data.append(serializable_item)
            json.dump(features_data, f, indent=4)
        print(f"Cơ sở dữ liệu đặc trưng đã được lưu vào: {path}")
    except IOError as e:
        print(f"Lỗi khi lưu file CSDL đặc trưng: {e}")
    except TypeError as e:
        print(f"Lỗi kiểu dữ liệu khi lưu CSDL (có thể do không chuyển đổi NumPy array): {e}")


def load_features_db(path=FEATURE_DB_PATH):
    """Tải cơ sở dữ liệu đặc trưng từ file JSON."""
    try:
        with open(path, 'r') as f:
            loaded_data_list = json.load(f)
            # Chuyển đổi list đặc trưng trở lại thành numpy arrays
            for video_entry in loaded_data_list:
                for segment in video_entry.get("segments", []):
                    if "features" in segment and isinstance(segment["features"], list):
                        segment["features"] = np.array(segment["features"], dtype=np.float32)
            print(f"Cơ sở dữ liệu đặc trưng (cấu trúc mới) đã được tải từ: {path}")
            return loaded_data_list
    except FileNotFoundError:
        print(f"Không tìm thấy file CSDL đặc trưng: {path}. Cần chạy quá trình indexing trước.")
        return []
    except json.JSONDecodeError:
        print(f"Lỗi giải mã JSON trong file: {path}. File có thể bị hỏng.")
        return []

def calculate_similarity(features1, features2, method='cosine'):
    """
    Tính toán độ tương đồng giữa hai vector đặc trưng.
    'cosine': 1 - cosine distance (càng gần 1 càng giống).
    'bhattacharyya': Cần đảm bảo features là histogram và tổng bằng 1. (càng gần 0 càng giống).
    'euclidean': Khoảng cách Euclidean (càng gần 0 càng giống).
    """
    if features1 is None or features2 is None:
        return 0 if method == 'cosine' else float('inf') # Hoặc một giá trị không tương đồng

    if features1.shape != features2.shape:
        print(f"Cảnh báo: Kích thước vector đặc trưng không khớp: {features1.shape} vs {features2.shape}")
        return 0 if method == 'cosine' else float('inf')

    if method == 'cosine':
        # scipy.spatial.distance.cosine trả về khoảng cách (1 - similarity)
        # Nên 1.0 - distance sẽ là similarity. Giá trị từ 0 đến 2.
        # Để chuẩn hơn, similarity = (v1 . v2) / (||v1|| * ||v2||)
        # Hoặc sử dụng 1 - cosine_distance nếu các vector đã chuẩn hóa (norm = 1)
        # Tuy nhiên, cosine distance từ scipy là 1 - (v1.v2 / (||v1||*||v2||))
        # Do đó, 1.0 - dist là một độ đo tương đồng tốt (càng gần 1 càng tốt)
        dist = cosine(features1, features2)
        if np.isnan(dist): # Xử lý trường hợp vector 0
            return 0
        return 1.0 - dist
    # elif method == 'bhattacharyya':
    #     # Khoảng cách Bhattacharyya yêu cầu các thành phần của histogram phải dương và tổng bằng 1.
    #     # Chúng ta đã chuẩn hóa histogram trong extract_hsv_histogram và extract_lbp_histogram
    #     # Tuy nhiên, vector tổng hợp có thể không còn tính chất này cho toàn bộ.
    #     # Tốt hơn là chia vector đặc trưng tổng hợp thành các phần (HSV, LBP, Edge) và tính riêng, rồi kết hợp.
    #     # Hoặc, nếu áp dụng cho toàn bộ, đảm bảo vector dương và chuẩn hóa lại (có thể làm thay đổi ý nghĩa).
    #     # Để đơn giản, ở đây ta giả sử có thể áp dụng được.
    #     # Bhattacharyya distance: càng nhỏ càng giống. Trả về khoảng cách.
    #     # Cần đảm bảo các vector đầu vào là phân phối xác suất (tổng bằng 1, không âm)
    #     # Nếu không, kết quả có thể không chính xác.
    #     # Đây là một điểm cần xem xét kỹ hơn khi chọn độ đo cho vector đặc trưng tổng hợp.
    #     # For simplicity, let's assume they are somewhat applicable or we use another metric.
    #     # A safer bet if not strictly histograms:
    #     print("Cảnh báo: Bhattacharyya thường dùng cho histogram. Kết quả có thể không tối ưu cho vector tổng hợp.")
    #     # Return a high distance if not applicable or use another part of the code to handle it.
    #     # For now, just compute it, but be cautious.
    #     if np.any(features1 < 0) or np.any(features2 < 0) or \
    #        not np.isclose(np.sum(features1), 1.0) or \
    #        not np.isclose(np.sum(features2), 1.0):
    #         # print("Bhattacharyya: Features are not valid probability distributions.")
    #         # Fallback or weighted average of component-wise bhattacharyya might be better
    #         pass # Allow computation but be aware
    #     return bhattacharyya(np.sqrt(features1), np.sqrt(features2)) # sqrt for Hellinger distance related to Bhattacharyya coeff
    #                                                              # This is often what's implemented.
    #                                                              # Result is a distance (smaller is better)
    elif method == 'euclidean':
        # Khoảng cách Euclidean: càng nhỏ càng giống.
        return euclidean(features1, features2)
    else:
        raise ValueError(f"Phương thức so sánh không được hỗ trợ: {method}")


def search_similar_videos(query_image_path, features_db, top_n=3):
    """
    Tìm kiếm các video tương đồng nhất với ảnh truy vấn.
    Args:
        query_image_path (str): Đường dẫn đến file ảnh truy vấn.
        features_db (list): Danh sách các dictionary, mỗi dict chứa 'video_path', 'keyframe_index', 'features'.
        top_n (int): Số lượng video tương đồng nhất cần trả về.

    Returns:
        list: Danh sách các tuple (video_path, max_similarity_score) được sắp xếp.
    """
    query_image = cv2.imread(query_image_path)
    if query_image is None:
        print(f"Lỗi: Không thể đọc ảnh truy vấn: {query_image_path}")
        return []

    query_features = extract_features_from_keyframe(query_image)
    if query_features is None:
        print("Lỗi: Không thể trích rút đặc trưng từ ảnh truy vấn.")
        return []

    video_similarities = {} # Lưu trữ điểm tương đồng cao nhất cho mỗi video

    for video_entry in features_db:
        # db_video_path = item['video_path']
        # db_keyframe_features = item['features']
        video_id = video_entry["video_id"]
        video_path_from_db = video_entry["video_path"]
        max_similarity_for_this_video = -1.0
        best_segment_info_for_this_video = None
        
        
        # Chọn phương thức so sánh. Cosine similarity thường tốt cho các vector đặc trưng dài.
        # Giá trị trả về từ 0 (khác nhau) đến 1 (giống nhau).
        # similarity_score = calculate_similarity(query_features, db_keyframe_features, method='cosine')
        for segment in video_entry.get("segments", []):
            db_segment_features = segment["features"] # Đã là NumPy array sau khi load
            
            similarity_score = calculate_similarity(query_features, db_segment_features, method='cosine')

            if similarity_score > max_similarity_for_this_video:
                max_similarity_for_this_video = similarity_score
                best_segment_info_for_this_video = segment # Lưu toàn bộ thông tin segment

        # Cập nhật điểm tương đồng cao nhất cho video này
        # if db_video_path not in video_similarities or similarity_score > video_similarities[db_video_path]:
        #     video_similarities[db_video_path] = similarity_score
        if best_segment_info_for_this_video is not None: # Nếu có segment nào đó giống
             # Lưu thông tin video và segment giống nhất của video đó
            video_similarities[video_id] = {
                "video_id": video_id,
                "video_path": video_path_from_db,
                "max_similarity_score": max_similarity_for_this_video,
                "best_segment_id": best_segment_info_for_this_video["segment_id"],
                "best_segment_start_time": best_segment_info_for_this_video["start_time_sec"],
                "best_segment_end_time": best_segment_info_for_this_video["end_time_sec"],
                "best_segment_keyframe_path": best_segment_info_for_this_video.get("keyframe_image_path", "N/A")
            }

    if not video_similarities:
        print("Không tìm thấy video nào trong CSDL hoặc không có sự tương đồng nào.")
        return []

    # Sắp xếp các video theo điểm tương đồng giảm dần
    sorted_videos = sorted(video_similarities.values(), key=lambda x: x["max_similarity_score"], reverse=True)

    return sorted_videos[:top_n]


if __name__ == '__main__':
    # --- Thử nghiệm ---
    # Giả sử features_database.json đã được tạo bởi main.py (chạy indexing trước)
    db = load_features_db()

    if db:
        # Chọn một ảnh truy vấn mẫu
        sample_query_dir = "../Dataset/QueryImages/"
        if not os.path.exists(sample_query_dir) or not os.listdir(sample_query_dir):
            print(f"Thư mục ảnh truy vấn mẫu '{sample_query_dir}' không tồn tại hoặc trống.")
            print("Vui lòng tạo thư mục và đặt một ảnh mẫu (ví dụ: 'query.jpg') vào đó để thử nghiệm.")
        else:
            query_image_files = [f for f in os.listdir(sample_query_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if query_image_files:
                query_image_file = os.path.join(sample_query_dir, query_image_files[0])
                print(f"\nĐang tìm kiếm với ảnh truy vấn: {query_image_file}")

                top_videos = search_similar_videos(query_image_file, db, top_n=3)

                if top_videos:
                    print("\nTop video tương đồng nhất:")
                    for video_path, score in top_videos:
                        print(f"- Video: {os.path.basename(video_path)}, Điểm tương đồng: {score:.4f}")
                else:
                    print("Không tìm thấy video tương đồng.")
            else:
                print(f"Không tìm thấy file ảnh nào trong '{sample_query_dir}'.")
    else:
        print("CSDL đặc trưng rỗng hoặc không thể tải. Vui lòng chạy quá trình indexing trước.")