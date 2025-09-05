import pickle
import cv2
import numpy as np
from kmeans_improvement import kmeans_silhouette
from save_keyframe import save_frames
from redundancy import redundancy
from logger import _setup_logger
import config
import time

logger = _setup_logger(__name__, config.LOG_LEVEL)

def scene_keyframe_extraction(scenes_path, features_path, video_path, save_path, folder_path):
    # Get lens segmentation data
    number_list = []
    with open(scenes_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # logger.debug(line)
            numbers = line.strip().split(' ')
            # logger.debug(numbers)
            number_list.extend([int(number) for number in numbers])
    
    logger.info(f"Đọc shot boundary thành công {scenes_path}")

    # Read inference data from local
    with open(features_path, 'rb') as file:
        features = pickle.load(file)

    features = np.asarray(features)
    logger.info(f"Đọc features thành công {features_path}")
    logger.info(f"Số lượng features {len(features)}")

    # Clustering at each shot to obtain keyframe sequence numbers
    keyframe_index = []
    start_total_extract = time.time()

    for i in range(0, len(number_list) - 1, 2):
        start = number_list[i]
        end = number_list[i + 1]
        logger.info(f""" {'#' * 100}
            Extract Keyframe On Shot {int(i/2 + 1)} ||| start: {start}, end: {end}
        """)
        sub_features = features[start:end]
        start_time = time.time()
        best_labels, best_centers, k, index = kmeans_silhouette(sub_features)
        end_time = time.time()
        logger.info(f"Thoi gian tim kiem tam cum: {end_time-start_time}")
        logger.debug(f"indices: {index}")
        final_index = [int(x + start) for x in index]
        # final_index.sort()
        logger.debug(f"clustering: {final_index}")
        logger.debug(f"segment start-end: {start}, {end}")
        final_index = redundancy(video_path, final_index, 0.94)
        logger.debug(f"filtered indices: {final_index}")
        keyframe_index += final_index

    end_total_extract = time.time()
    logger.info(f"Thời gian extract keyframe: {end_total_extract-start_total_extract}")

    keyframe_index.sort()
    logger.debug(f"final_index: {keyframe_index}")

    # save keyframe
    save_frames(keyframe_index, video_path, save_path, folder_path)
    return keyframe_index


