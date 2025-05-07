# embedding video func
import numpy as np
import cv2

def extract_features_from_image(image):
    feature_vector = np.random.rand(-1, 1, 128) # demo
    return(feature_vector)

def extract_features_from_video(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval_seconds = 0.2 # Define the interval between frames to extract features
    frame_interval = int(fps * frame_interval_seconds)

    features = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            feature_vector = extract_features_from_image(frame)
            features.append(feature_vector)
        frame_count += 1

    cap.release()
    return np.array(features)