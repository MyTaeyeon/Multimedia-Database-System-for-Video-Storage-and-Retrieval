# app.py
import streamlit as st
# from search_engine.search import load_video_features, find_top_videos
# from search_engine.extract_features import extract_features_from_image  # Giả sử bạn có rồi
import os

FEATURES_DIR = 'data/features'
VIDEOS_DIR = 'data/videos'

st.title("Video Search by Image")

uploaded_image = st.file_uploader("Upload a query image", type=['jpg', 'png'])

if uploaded_image:
    print("yeb!")
    # query_feature = extract_features_from_image(uploaded_image)
    # video_features = load_video_features(FEATURES_DIR)
    # top_videos = find_top_videos(query_feature, video_features)

    # st.write("Top 3 similar videos:")
    # for video_name, score in top_videos:
    #     video_path = os.path.join(VIDEOS_DIR, f"{video_name}.mp4")
    #     st.video(video_path)
    #     st.caption(f"Similarity score: {score:.4f}")
