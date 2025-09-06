import numpy as np
from scipy.spatial.distance import cdist
from init_center import kmeans_init
from logger import _setup_logger
import config
import torch
import cupy
from cuml.metrics.cluster import silhouette_score

print(torch.__version__)
print(torch.cuda.is_available())     # True nếu có GPU CUDA
print(torch.cuda.device_count())     # số lượng GPU
print(torch.cuda.get_device_name(0)) # tên GPU đầu tiên



logger = _setup_logger(__name__, config.LOG_LEVEL)

def gpu_cdist(arr, p=2):
    """
    Tính ma trận khoảng cách giữa các phần tử trong mảng 1D bằng GPU (PyTorch).

    Parameters
    ----------
    arr : array-like (list, np.ndarray, torch.Tensor)
        Mảng 1 chiều chứa các số (sẽ được coi là vector 1D).
    p : float (default=2)
        Norm để tính khoảng cách (p=2 = Euclid, p=1 = Manhattan, ...)

    Returns
    -------
    np.ndarray
        Ma trận khoảng cách (n x n).
    """
    # Nếu arr không phải tensor thì chuyển sang torch.Tensor
    if not isinstance(arr, torch.Tensor):
        t = torch.as_tensor(arr, dtype=torch.float64, device="cuda")
    else:
        # Nếu là tensor nhưng không ở CUDA thì chuyển sang
        if arr.device.type != "cuda":
            t = arr.to(torch.float64).to("cuda")
        else:
            t = arr.to(torch.float64)

    # print("Tensor shape:", t.shape, "Device:", t.device)

    # Tính khoảng cách bằng torch.cdist
    D = torch.cdist(t, t, p=p)

    # Trả về dưới dạng numpy array (CPU)
    return D

def get_min_distance_idx(D: torch.Tensor, restore_diag: bool = True):
    """
    Tìm cặp (i, j) có khoảng cách nhỏ nhất trong ma trận D,
    bỏ qua đường chéo.

    Parameters
    ----------
    D : torch.Tensor
        Ma trận khoảng cách vuông (k x k).
    restore_diag : bool, default=True
        Nếu True, sao lưu và khôi phục lại đường chéo sau khi tính.
        Nếu False, đường chéo sẽ bị thay thế = +inf vĩnh viễn.

    Returns
    -------
    (i, j, val) : tuple
        i, j là chỉ số của cặp có khoảng cách nhỏ nhất.
        val là khoảng cách nhỏ nhất.
    """
    diag = D.diagonal()

    if restore_diag:
        backup = diag.clone()
        diag.fill_(float('inf'))
        idx = torch.argmin(D)
        diag.copy_(backup)  # khôi phục
    else:
        diag.fill_(float('inf'))
        idx = torch.argmin(D)

    i, j = divmod(idx.item(), D.size(1))
    return i, j

def update_cluster_centers_fully_vectorized(features, clusters, k):
    # Ensure inputs are torch tensors
    
    # Compute cluster means using scatter_add
    counts = torch.bincount(clusters, minlength=k-1).float().clamp(min=1)  # Avoid division by zero
    logger.debug(f"counts: {counts}")
    cluster_means = torch.zeros(k-1, features.shape[1], device=features.device, dtype=features.dtype)
    cluster_means.scatter_add_(0, clusters.unsqueeze(1).expand(-1, features.shape[1]), features)
    cluster_means /= counts.view(-1, 1)
    logger.debug(f"cluster_means: {cluster_means}")

    
    # Compute distances to respective cluster means
    cluster_indices = torch.arange(k-1, device=features.device)
    logger.debug(f"cluster_indices: {cluster_indices}")
    mask = (clusters[:, None] == cluster_indices[None, :]).float()
    logger.debug(f"mask {mask}")
    
    distances = torch.norm(features[:, None, :] - cluster_means[None, :, :], dim=2)
    logger.debug(f"distances: {distances}")
    distances = distances * mask + (1 - mask) * float('9999999999999999')  # Mask out non-cluster points
    # distances = distances * mask  # Mask out non-cluster points
    logger.debug(f"distances: {distances}")
    
    
    # Find closest point indices
    closest_indices = torch.argmin(distances, dim=0)
    logger.debug(f"closest_indices: {closest_indices}")
    new_centers = features[closest_indices]
    
    return new_centers

def kmeans_silhouette(features):
    # calculate sqrt(n)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using device: {device}")
    features = torch.as_tensor(features, dtype=torch.float64, device=device)

    sqrt_n = int(torch.sqrt(torch.tensor(len(features), 
                                        dtype=torch.float64, 
                                        device=device)).item())

    # Initialise k and clustering results
    k = sqrt_n
    best_k = k
    best_clusters = None
    best_avg_silhouette = -1

    logger.debug("Khởi tạo biến ban đầu thành công")
    logger.debug(f"k: {k}")

    # Results of the selection of the initial center (cần phiên bản torch)
    clusters, centers = kmeans_init(features)


    best_clusters = clusters
    best_centers = centers
    matches = (best_centers[:, None, :] == features[None, :, :]).all(dim=2)  # [K, N]

    # Lấy index đầu tiên True cho mỗi center
    center_indices = matches.float().argmax(dim=1)  # [K]

    # đảm bảo kiểu long
    center_indices = center_indices.to(dtype=torch.long, device=features.device)
    
    logger.debug(f"Cập nhật cụm và tâm cụm tốt nhất thành công")
    logger.debug(f"best_clusters: {best_clusters}")
    logger.debug(f"best_centers: {best_centers} (len={len(best_centers)})")
    logger.debug(f"best_center_index: {center_indices}")



    while k > 2:
        logger.debug(f"{'-' * 50}")
        # Calculate the Euclidean distance between cluster centers
        cluster_centers = centers

        distances = gpu_cdist(cluster_centers)
        logger.debug(f"distances: {distances}")

        # Find the indexes of the two the nearest clusters
        merge_cluster_indices = get_min_distance_idx(distances)

        logger.debug(f"merge_cluster_indices: {merge_cluster_indices}")

        # Merge the two the nearest clusters and change the high cluster number to the low cluster number
        merged_cluster = torch.where(
            clusters == merge_cluster_indices[1],
            torch.tensor(merge_cluster_indices[0], device=clusters.device, dtype=clusters.dtype),
            clusters
        )

        # Bước 2: reindex lại cụm, các nhãn > j thì giảm 1
        clusters = torch.where(
            merged_cluster > merge_cluster_indices[1],
            merged_cluster - 1,
            merged_cluster
        )

        logger.debug(f"clusters sau khi gộp: {clusters}")

        # Update the cluster center, selecting the actual data point as the new cluster center
        new_centers = update_cluster_centers_fully_vectorized(features, clusters, k)

        logger.debug(f"new_centers: {new_centers}")

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
            best_clusters = clusters
            best_centers = centers
            matches = (best_centers[:, None, :] == features[None, :, :]).all(dim=2)  # [K, N]

            # Lấy index đầu tiên True cho mỗi center
            center_indices = matches.float().argmax(dim=1)  # [K]

            # đảm bảo kiểu long
            center_indices = center_indices.to(dtype=torch.long, device=features.device)
            
        logger.debug(f"Cập nhật cụm và tâm cụm tốt nhất thành công")
        logger.debug(f"best_clusters: {best_clusters}")
        logger.debug(f"best_centers: {best_centers} (len={len(best_centers)})")
        logger.debug(f"best_center_index: {center_indices}")
        logger.debug(f"{'-' * 50}")

    # return result
    logger.debug(f"Phân cụm và tâm cụm tốt nhất thành công")
    logger.debug(f"best_k: {best_k}")
    logger.debug(f"best_clusters: {best_clusters}")
    logger.debug(f"best_centers: {best_centers} (len={len(best_centers)})")
    logger.debug(f"best_center_index: {center_indices}")

    return best_clusters, best_centers, best_k, center_indices

