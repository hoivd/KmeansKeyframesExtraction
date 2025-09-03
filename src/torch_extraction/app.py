from keyframe_extraction import scene_keyframe_extraction
import json
import os
from logger import _setup_logger
import config

logger = _setup_logger(__name__, config.LOG_LEVEL)

def main():
    # ====== CHỈNH ĐƯỜNG DẪN ======
    video_path = "D:/Keyframe-Extraction-for-video-summarization/src/data/videos/L21_V005.mp4"
    scenes_path = "D:/Keyframe-Extraction-for-video-summarization/src/data/shots_boundary/L21_V005.txt"
    features_path = "D:/Keyframe-Extraction-for-video-summarization/src/data/embeddings/L21_V005_features.pkl"

    # Nơi lưu keyframes
    save_path = "D:/Keyframe-Extraction-for-video-summarization/src/output"
    folder_path = "video_keyframes"

    os.makedirs(save_path, exist_ok=True)

    keyframe_index = scene_keyframe_extraction(
        scenes_path=scenes_path,
        features_path=features_path,
        video_path=video_path,
        save_path=save_path,
        folder_path=folder_path
    )

    keyframes_index_file = 'D:/Keyframe-Extraction-for-video-summarization/src/output/L21_V005_keyframes_index.json'

    # Lưu mảng vào file JSON
    with open(keyframes_index_file, 'w') as json_file:
        json.dump(keyframe_index, json_file)

    print(f'Frameindexs đã được lưu vào file {keyframes_index_file}')

    print(f"✅ Keyframes đã lưu trong {os.path.join(save_path, folder_path)}")

if __name__ == "__main__":
    main()