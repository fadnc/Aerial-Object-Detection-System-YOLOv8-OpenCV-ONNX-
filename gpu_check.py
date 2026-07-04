import torch

available = torch.cuda.is_available()
print(f"CUDA available: {available}")

if available:
    print(f"Device: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected - training/inference will run on CPU (slower).")
