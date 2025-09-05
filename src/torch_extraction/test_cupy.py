import cupy as cp

# Kiểm tra phiên bản CuPy
print("CuPy version:", cp.__version__)

# Kiểm tra xem có GPU khả dụng không
try:
    cp.cuda.Device(0).compute_capability  # gọi device đầu tiên
    print("GPU detected:", cp.cuda.runtime.getDeviceProperties(0)["name"])
except Exception as e:
    print("Không tìm thấy GPU hoặc CuPy chưa kết nối CUDA:", e)

# Thử tính toán đơn giản trên GPU
x = cp.arange(10**6, dtype=cp.float32)
y = cp.arange(10**6, dtype=cp.float32)
z = x + y
print("Tính toán thành công, kết quả mẫu:", z[:10])
