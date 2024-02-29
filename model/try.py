import torch
from sample import *

B, N, K= 1, 10, 3
# 创建一个形状为 [B, N, 3] 的 points 张量
points = torch.randn(B, N, 3)
idx_k = FPS(points, K)
indice = torch.arange(B).view(B, 1).repeat(1, K)
k_points = points[indice, idx_k, :]
for i in range(N):
    print(points[0][i])

print()
for i in range(K):
    print(k_points[0][i])