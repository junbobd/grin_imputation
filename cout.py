import torch

# 加载模型文件（查看 checkpoint 内部结构的小工具）
ckpt = torch.load("best_grin.final.pt", map_location="cpu")

# 看里面有什么 key
print(ckpt.keys())