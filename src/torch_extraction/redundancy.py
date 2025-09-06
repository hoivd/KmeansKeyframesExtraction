import av
import numpy as np
import torch

def redundancy(video_path, keyframe_index, threshold, device="cuda"):
    # --- 1. Hàm con tính histogram ---
    def color_histogram(img):
        import cv2
        hist = cv2.calcHist([img], [0, 1, 2], None,
                            [8, 8, 8], [0, 255, 0, 255, 0, 255])
        return hist.flatten()

    # --- 2. Đọc keyframes bằng PyAV ---
    histograms = []
    container = av.open(video_path)
    stream = container.streams.video[0]

    keyframe_index_set = set(keyframe_index)
    frames = {}
    for frame in container.decode(stream):
        if frame.index in keyframe_index_set:
            img = frame.to_ndarray(format="bgr24")
            frames[frame.index] = color_histogram(img)
        if len(frames) == len(keyframe_index):
            break
    container.close()

    # sắp xếp đúng thứ tự input
    histograms = [frames[idx] for idx in keyframe_index if idx in frames]
    histograms = np.array(histograms)

    # --- 3. Lọc frame ít thông tin ---
    mask = np.sum(histograms > 0, axis=1) > 10
    new_histogram = histograms[mask]
    mid_index = np.array(keyframe_index)[mask]

    if len(new_histogram) == 0:
        return [keyframe_index[0]]

    # --- 4. Chuyển sang Torch & normalize ---
    H = torch.tensor(new_histogram, dtype=torch.float32, device=device)
    H = torch.nn.functional.normalize(H, p=2, dim=1)  # shape (N, 512)

    # --- 5. Tính cosine similarity toàn bộ ---
    simis = H @ H.T   # shape (N, N)

    # --- 6. Tạo mask upper-triangular (so sánh i<j) ---
    N = simis.size(0)
    tri_mask = torch.triu(torch.ones((N, N), device=device, dtype=torch.bool), diagonal=1)

    # --- 7. Tìm các cặp trùng lặp (similarity > threshold) ---
    dup_pairs = (simis > threshold) & tri_mask

    # --- 8. Xác định index bị loại ---
    del_mask = torch.any(dup_pairs, dim=0)

    # --- 9. Lấy final index ---
    keep_mask = ~del_mask
    final_index = mid_index[keep_mask.cpu().numpy()]
    final_index = sorted(final_index.tolist())

    return final_index
