import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print("PyTorch CUDA support:", torch.version.cuda) 
print("cuDNN version:", torch.backends.cudnn.version())
# True
# 1
# NVIDIA GeForce GTX 1650 with Max-Q Design
# PyTorch CUDA support: 11.8
# cuDNN version: 90100