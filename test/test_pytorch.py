import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    
    # 测试GPU计算
    x = torch.rand(3, 3).cuda()
    print(f"\nGPU上的张量:\n{x}")
    print("✅ GPU加速可用！")
else:
    print("⚠️ GPU不可用，使用CPU模式")