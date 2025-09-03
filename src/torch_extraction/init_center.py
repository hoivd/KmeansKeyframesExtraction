import numpy as np
from tqdm import tqdm
from logger import _setup_logger
import config

logger = _setup_logger(__name__, config.LOG_LEVEL)

def kmeans_init(data):
    logger.debug("🔹 In the process of initialising the center")
    n = len(data)
    sqrt_n = int(np.sqrt(n))           # số tâm cần chọn
    centers = []
    label = []

    # pick init_center
    while len(centers) < sqrt_n:
        sse_min = float('inf')
        join_center = data[0]

        # tqdm để quan sát tiến trình duyệt qua n điểm
        for i in tqdm(range(n), desc=f"Selecting center {len(centers)+1}/{sqrt_n}"):
            center = centers.copy()
            
            # kiểm tra tránh chọn lại tâm đã có
            if len(centers) == 0 or not np.any(np.all(data[i] == centers, axis=1)):
                center.append(data[i])
                center = np.array(center)
                sse = 0.0

                # Cluster operation
                cluster_labels = np.zeros(len(data)).astype(int)
                for k in range(len(data)):
                    distances = [np.sqrt(np.sum((data[k] - cen) ** 2)) for cen in center]
                    nearest_cluster = np.argmin(distances)
                    cluster_labels[k] = nearest_cluster

                # Based on the results of the cluster operation, calculate sse
                for j in range(len(center)):
                    cluster_points = [data[l] for l in range(len(cluster_labels)) if cluster_labels[l] == j]
                    singe_sse = sum(np.linalg.norm(point - center[j]) for point in cluster_points)
                    sse += singe_sse

                if sse < sse_min:
                    sse_min = sse
                    join_center = data[i]
                    label = cluster_labels.copy()

        centers.append(join_center)
    
    clusters = np.array(label) 
    centers = np.array(centers)

    logger.info(f"Khởi tạo cụm và tâm cụm ban đầu thành công")
    logger.debug(f"Các cụm ban đầu {clusters}")
    logger.debug(f"Các tâm ban đầu {centers}")

    return clusters, centers

