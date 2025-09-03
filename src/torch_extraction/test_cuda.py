import torch
print(torch.__version__)          # version PyTorch
print(torch.version.cuda)         # version CUDA build-in
print(torch.cuda.is_available())  # True nếu nhận GPU
print(torch.cuda.get_device_name(0))
