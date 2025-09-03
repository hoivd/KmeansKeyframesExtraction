import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from init_center import kmeans_init
from logger import _setup_logger
import config

logger = _setup_logger(__name__, config.LOG_LEVEL)

def kmeans_silhouette(features):
    # calculate sqrt(n)
    sqrt_n = int(np.sqrt(len(features)))
    # Initialise k and clustering results
    k = sqrt_n
    best_k = k
    best_clusters = None
    best_avg_silhouette = -1

    logger.info(f"Khởi tạo biến ban đầu thành công")
    logger.debug(f"k: {k}")

    # Results of the selection of the initial center
    clusters, centers = kmeans_init(features)
    # Iterative Procedur    

    best_clusters = clusters.copy()
    best_centers = centers.copy()
    center_indices = []
    for cluster_center in best_centers:
        center_index = np.where((features == cluster_center).all(axis=1))[0][0]
        center_indices.append(center_index)
    
    logger.info(f"Cập nhật cụm và tâm cụm tốt nhất thành công")
    logger.debug(f"best_clusters: {best_clusters}")
    logger.debug(f"best_centers: {best_centers} (len={len(best_centers)})")
    logger.debug(f"best_center_index: {center_indices}")

    while k > 2:
        logger.info(f"{'-' * 50}")
        # Calculate the Euclidean distance between cluster centers
        cluster_centers = centers

        distances = cdist(cluster_centers, cluster_centers, metric='euclidean')
        logger.debug(f"distances: {distances}")

        # Find the indexes of the two the nearest clusters
        min_distance = np.inf
        merge_cluster_indices = None
        # Iterate over the values in the upper right corner of the matrix
        for i in range(k):
            for j in range(i + 1, k):
                if distances[i, j] < min_distance:
                    min_distance = distances[i, j]
                    merge_cluster_indices = (i, j)

        # Merge the two the nearest clusters and change the high cluster number to the low cluster number
        merged_cluster = np.where(clusters == merge_cluster_indices[1], merge_cluster_indices[0], clusters)

        # Update clustering results
        clusters = np.where(merged_cluster > merge_cluster_indices[1], merged_cluster - 1, merged_cluster)

        # Update the cluster center, selecting the actual data point as the new cluster center
        new_centers = []
        for cluster_id in range(k - 1):
            # Get samples of the current cluster
            cluster_samples = features[clusters == cluster_id]
            # Calculate the current cluster mean
            cluster_mean = np.mean(cluster_samples, axis=0)
            # Calculate the Euclidean distance between the sample and the centre point to find the actual center
            distances = np.linalg.norm(cluster_samples - cluster_mean, axis=1)
            closest_sample_index = np.argmin(distances)
            # Choose the nearest sample as the new cluster centroid
            new_centers.append(cluster_samples[closest_sample_index])

        centers = new_centers
        # update number of cluster
        k -= 1
        # Calculate Silhouette Coefficient
        avg_silhouette = silhouette_score(features, clusters)

        logger.debug(f"avg_silhouette: {avg_silhouette}")
        logger.debug(f"best_avg_silhouette: {best_avg_silhouette}")

        # 更新最佳结果
        if avg_silhouette > best_avg_silhouette:
            best_avg_silhouette = avg_silhouette
            best_k = k
            best_clusters = clusters.copy()
            best_centers = centers.copy()
            center_indices = []
            for cluster_center in best_centers:
                center_index = np.where((features == cluster_center).all(axis=1))[0][0]
                center_indices.append(center_index)
        logger.info(f"Cập nhật cụm và tâm cụm tốt nhất thành công")
        logger.debug(f"best_clusters: {best_clusters}")
        logger.debug(f"best_centers: {best_centers} (len={len(best_centers)})")
        logger.debug(f"best_center_index: {center_indices}")
        logger.info(f"{'-' * 50}")

    # return result
    logger.info(f"Phân cụm và tâm cụm tốt nhất thành công")
    logger.debug(f"best_k: {best_k}")
    logger.debug(f"best_clusters: {best_clusters}")
    logger.debug(f"best_centers: {best_centers} (len={len(best_centers)})")
    logger.debug(f"best_center_index: {center_indices}")

    return best_clusters, best_centers, best_k, center_indices

