import numpy as np
from tqdm import tqdm
from logger import _setup_logger
import config
import torch

logger = _setup_logger(__name__, config.LOG_LEVEL)

def kmeans_init(data: torch.Tensor, device="cuda"):
    """
    Khởi tạo tâm cụm theo cách greedy minimization SSE
    data: torch.Tensor [N, D], dtype=float64
    device: "cuda" hoặc "cpu"

    return:
        clusters: torch.LongTensor [N] (nhãn cụm)
        centers:  torch.Float64Tensor [K, D] (tâm cụm)
    """

    logger.debug("🔹 In the process of initialising the center")

    data = data.to(device, dtype=torch.float64)   # đảm bảo data ở GPU float64
    n = data.shape[0]
    sqrt_n = int(torch.sqrt(torch.tensor(n, dtype=torch.float64, device=device)).item())

    centers = []
    label = None

    # pick init_center
    while len(centers) < sqrt_n:
        sse_min = float("inf")
        join_center = data[0]

        # tqdm để quan sát tiến trình duyệt qua n điểm
        for i in tqdm(range(n), desc=f"Selecting center {len(centers)+1}/{sqrt_n}"):

            # copy các tâm hiện tại
            if len(centers) > 0:
                current_centers = torch.stack(centers, dim=0)  # [m, D]
                # kiểm tra tránh chọn lại tâm đã có
                if torch.any(torch.all(data[i] == current_centers, dim=1)):
                    continue
                center = torch.cat([current_centers, data[i].unsqueeze(0)], dim=0)
            else:
                center = data[i].unsqueeze(0)

            # ---- phân cụm tạm thời ----
            distances = torch.cdist(data, center, p=2)  # [N, m+1]
            cluster_labels = torch.argmin(distances, dim=1)

            # ---- tính SSE ----
            min_distances = distances[torch.arange(n, device=device), cluster_labels]
            sse = min_distances.sum().item()

            if sse < sse_min:
                sse_min = sse
                join_center = data[i]
                label = cluster_labels.clone()

        centers.append(join_center)

    # Chuyển về tensor torch luôn
    clusters = label  # torch.LongTensor [N]
    centers = torch.stack(centers, dim=0)  # torch.Float64Tensor [sqrt_n, D]

    logger.info("Khởi tạo cụm và tâm cụm ban đầu thành công")
    logger.debug(f"Các cụm ban đầu {clusters}")
    logger.debug(f"Các tâm ban đầu {centers}")

    return clusters, centers

