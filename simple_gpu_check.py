import torch

# Check PyTorch version and CUDA
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0))
    capability = torch.cuda.get_device_capability(0)
    print(f"Compute capability: {capability[0]}.{capability[1]}")
    
    # Check if RTX 5060 Ti uses sm_120 or sm_90
    major, minor = capability
    print(f"Architecture code: sm_{major}{minor}")
    
    if major >= 12:
        print("✅ Using sm_120+ (Blackwell architecture)")
    elif major >= 9:
        print("✅ Using sm_90+ (Ada Lovelace architecture)")
    else:
        print(f"⚠️ Using older architecture: sm_{major}{minor}")
