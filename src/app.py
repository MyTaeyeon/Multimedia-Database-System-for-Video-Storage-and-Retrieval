import os
import time
import multiprocessing

# Import các module cần thiết
from utils.video_segmentation import segment_video_into_shots
from utils.feature_extraction import extract_features_from_keyframe
from utils.search import save_features_db, load_features_db, search_similar_videos

# --- Đường dẫn cấu hình ---
VIDEO_DATASET_PATH = "./inputs/videos/"
QUERY_IMAGE_PATH = "./inputs/images/" # Thư mục chứa ảnh truy vấn
OUTPUT_FEATURES_DB = "./outputs/extracted_features/features_database.json"
OUTPUT_KEYFRAME_DIR = "./outputs/keyframes/" # Thư mục lưu keyframes

# Tạo các thư mục nếu chưa có
os.makedirs(os.path.dirname(OUTPUT_FEATURES_DB), exist_ok=True)
os.makedirs(VIDEO_DATASET_PATH, exist_ok=True)
os.makedirs(QUERY_IMAGE_PATH, exist_ok=True)
os.makedirs(OUTPUT_KEYFRAME_DIR, exist_ok=True)

def process_single_video_for_db(video_path_tuple): # Hàm này phải ở top-level hoặc pickle được
    video_path, video_folder_path = video_path_tuple # Giải nén tuple
    video_file = os.path.basename(video_path)
    video_id = os.path.splitext(video_file)[0]
    print(f"Worker {os.getpid()} bắt đầu xử lý video: {video_file}")
    video_features = []
    # 1. Phân đoạn video và trích xuất keyframes
    processed_shots_from_segmentation, fps = segment_video_into_shots(video_path,keyframe_save_dir=OUTPUT_KEYFRAME_DIR) # Giả sử video_segmentation được import

    if not processed_shots_from_segmentation:
        print(f"Không trích xuất được shot nào từ {video_file} bởi worker {os.getpid()}.")
        return None # Trả về None nếu không có shot

    video_data_entry = {
        "video_id": video_id,
        "video_path": video_path,
        "fps": fps,
        "segments": []
    }

    for shot_info_from_seg in processed_shots_from_segmentation:
        keyframe_image = shot_info_from_seg["keyframe_image_for_feature"]
        features = extract_features_from_keyframe(keyframe_image)

        if features is not None:
            print("hehe")
            segment_entry = {
                "segment_id": shot_info_from_seg["segment_id"],
                "start_time_sec": shot_info_from_seg["start_time_sec"],
                "end_time_sec": shot_info_from_seg["end_time_sec"],
                "keyframe_image_path": shot_info_from_seg["keyframe_image_path"], # Lưu lại đường dẫn
                "features": features.tolist() # Chuyển NumPy array sang list để lưu JSON
            }
            video_data_entry["segments"].append(segment_entry)
        else:
            print(f"Không thể trích xuất đặc trưng cho {shot_info_from_seg['segment_id']} của video {video_file}")
    
    if not video_data_entry["segments"]: # Nếu không có segment nào có feature
        return None

    print(f"Worker {os.getpid()} xử lý xong video {video_file}, trích xuất {len(video_data_entry['segments'])} segments với features.")
    return video_data_entry

def build_feature_database_parallel(video_folder_path):
    all_video_data_entries = []
    video_files_full_path = [os.path.join(video_folder_path, f)
                   for f in os.listdir(video_folder_path)
                   if os.path.isfile(os.path.join(video_folder_path, f)) and
                   f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files_full_path:
        print(f"Không tìm thấy file video nào trong thư mục: {video_folder_path}")
        return

    print(f"Bắt đầu quá trình xây dựng CSDL song song cho {len(video_files_full_path)} videos...")

    # Tạo một tuple (video_path, video_folder_path) cho mỗi video
    # video_folder_path có thể không cần thiết nếu các module khác không dùng
    tasks = [(path, video_folder_path) for path in video_files_full_path]

    # Sử dụng số lượng CPU cores hợp lý, ví dụ: os.cpu_count() - 1 hoặc một số cố định
    num_processes = min(max(1, multiprocessing.cpu_count() -1), 4) # Giới hạn max 4 processes để ví dụ
    print(f"Sử dụng {num_processes} processes...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        # map sẽ áp dụng hàm process_single_video_for_db cho từng item trong video_files_full_path
        # và trả về một list các kết quả (list của các list features)
        results = pool.map(process_single_video_for_db, tasks)

    # Gom kết quả từ các worker
    for video_entry in results:
        if video_entry:
            all_video_data_entries.append(video_entry)

    if all_video_data_entries:
        save_features_db(all_video_data_entries, OUTPUT_FEATURES_DB)
    else:
        print("Không có đặc trưng nào được trích xuất để lưu.")
    print("\nQuá trình xây dựng CSDL đặc trưng song song hoàn tất.")


def build_feature_database(video_folder_path):
    """
    Xây dựng cơ sở dữ liệu đặc trưng từ tất cả các video trong một thư mục.
    """
    all_features_data = []
    video_files = [f for f in os.listdir(video_folder_path)
                   if os.path.isfile(os.path.join(video_folder_path, f)) and
                   f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files:
        print(f"Không tìm thấy file video nào trong thư mục: {video_folder_path}")
        return

    print(f"Bắt đầu quá trình xây dựng CSDL đặc trưng cho {len(video_files)} videos...")

    for video_file in video_files:
        video_path = os.path.join(video_folder_path, video_file)
        print(f"\nĐang xử lý video: {video_file}")
        start_time_video = time.time()

        # 1. Phân đoạn video và trích xuất keyframes
        keyframes = segment_video_into_shots(video_path,keyframe_save_dir=OUTPUT_KEYFRAME_DIR)

        if not keyframes:
            print(f"Không trích xuất được keyframe nào từ {video_file}.")
            continue

        # 2. Trích rút đặc trưng từ mỗi keyframe
        for i, kf_image in enumerate(keyframes):
            if kf_image is None:
                print(f"Cảnh báo: Keyframe {i} của video {video_file} bị rỗng.")
                continue
            features = extract_features_from_keyframe(kf_image)
            if features is not None:
                all_features_data.append({
                    'video_path': video_path, # Lưu đường dẫn đầy đủ hoặc tương đối
                    'video_name': video_file, # Chỉ tên file
                    'keyframe_index_in_video': i, # Thứ tự keyframe trong video đó
                    'features': features # Vector đặc trưng NumPy
                })
            else:
                print(f"Không thể trích xuất đặc trưng cho keyframe {i} của video {video_file}")

        end_time_video = time.time()
        print(f"Xử lý xong video {video_file} trong {end_time_video - start_time_video:.2f} giây. Trích xuất được {len(keyframes)} keyframes.")

    # 3. Lưu trữ CSDL đặc trưng
    if all_features_data:
        save_features_db(all_features_data, OUTPUT_FEATURES_DB)
    else:
        print("Không có đặc trưng nào được trích xuất để lưu.")

    print("\nQuá trình xây dựng CSDL đặc trưng hoàn tất.")


def run_search_on_sample_query(query_folder_path, top_n_results=3):
    features_db = load_features_db(OUTPUT_FEATURES_DB) 
    if not features_db:
        print("CSDL đặc trưng (cấu trúc mới) rỗng hoặc không thể tải. Vui lòng chạy indexing trước ('build').")
        return
    
    query_image_files = [f for f in os.listdir(query_folder_path)
                         if os.path.isfile(os.path.join(query_folder_path, f)) and
                         f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not query_image_files:
        print(f"Không tìm thấy ảnh truy vấn nào trong thư mục: {query_folder_path}")
        return

    for query_img_file in query_image_files:
        full_query_path = os.path.join(query_folder_path, query_img_file)
        print(f"\n--- Đang tìm kiếm cho ảnh: {query_img_file} ---")
        start_time_search = time.time()

        top_videos_with_segments = search_similar_videos(full_query_path, features_db, top_n=top_n_results)

        end_time_search = time.time()
        print(f"Tìm kiếm hoàn tất trong {end_time_search - start_time_search:.2f} giây.")

        if top_videos_with_segments:
            print(f"Top {len(top_videos_with_segments)} video tương đồng nhất:")
            for video_info in top_videos_with_segments: # video_info giờ là dict
                print(f"  - Video ID: {video_info['video_id']} (Path: {video_info['video_path']})")
                print(f"    Best Segment ID: {video_info['best_segment_id']}")
                print(f"    Segment Time: {video_info['best_segment_start_time']:.2f}s - {video_info['best_segment_end_time']:.2f}s")
                print(f"    Similarity Score: {video_info['max_similarity_score']:.4f}")
                print(f"    Keyframe of best segment: {video_info['best_segment_keyframe_path']}") # Thêm dòng này
        else:
            print("Không tìm thấy video tương đồng.")
        print("--- Kết thúc tìm kiếm ---")


if __name__ == "__main__":
    print("Hệ thống Tìm kiếm Video theo Nội dung Phong cảnh")
    print("-----------------------------------------------")
    print("Các lệnh có sẵn: build, search, exit")

    # Kiểm tra xem có đủ video trong dataset không
    video_files_in_dataset = [f for f in os.listdir(VIDEO_DATASET_PATH)
                              if os.path.isfile(os.path.join(VIDEO_DATASET_PATH, f)) and
                              f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if len(video_files_in_dataset) < 1: # Đặt yêu cầu tối thiểu, ví dụ 1 video
        print(f"\nCẢNH BÁO: Thư mục video '{VIDEO_DATASET_PATH}' chứa ít hơn 1 video.")
        print("Vui lòng thêm ít nhất 1 video (khuyến nghị 100 video theo yêu cầu) vào thư mục này.")
        # Nếu bạn muốn bắt buộc 100 video, thay 1 bằng 100

    while True:
        command = input("\nNhập lệnh (build/search/exit): ").strip().lower()

        if command == 'build':
            if not video_files_in_dataset:
                 print(f"Không có video trong {VIDEO_DATASET_PATH} để xây dựng CSDL. Vui lòng thêm video.")
            else:
                build_feature_database_parallel(VIDEO_DATASET_PATH)
        elif command == 'search':
            if not os.path.exists(OUTPUT_FEATURES_DB):
                print("CSDL đặc trưng chưa được xây dựng. Vui lòng chạy lệnh 'build' trước.")
            elif not os.listdir(QUERY_IMAGE_PATH):
                print(f"Thư mục ảnh truy vấn '{QUERY_IMAGE_PATH}' đang trống. Vui lòng thêm ảnh để tìm kiếm.")
            else:
                run_search_on_sample_query(QUERY_IMAGE_PATH)
        elif command == 'exit':
            print("Đang thoát chương trình.")
            break
        else:
            print("Lệnh không hợp lệ. Vui lòng nhập 'build', 'search', hoặc 'exit'.")