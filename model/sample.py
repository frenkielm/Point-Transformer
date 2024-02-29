import torch
def sample_k(x, index):
    """
    x: B * N * F
    index : B * N * K
    sample : B * N * K * F
    """
    B, N, F = x.shape
    _, _, K = index.shape
    indice = torch.arange(B).view(B, 1, 1).repeat(1, N, K) # (B, N, K) 数据为batch的索引
    sample = x[indice, index, :] # 用indice的数据做x第一个维度的索引，indice的结构做sample的前三维，同时是index的索引，:是每个点的特征向量
    return sample

def compute_kidx(xyz, k_num):
    """
    xyz: B * N * 3
    return:
    k_idx : B * N * K
    k_dist : B * N * K * 3 用于计算位置编码
    """
    B, N, _ = xyz.shape
    K = k_num
    idx = torch.arange(N).view(1, 1, N).repeat(B, N, 1)
    indice = torch.arange(B).view(B, 1, 1).repeat(1, N, N) # 数据为B
    nxyz = xyz[indice, idx, :] # B * N * N * 3
    xyz_plus = xyz.view(B, N, 1, 3).repeat(1, 1, N, 1) # B * N * N * 3
    dist = torch.sqrt(torch.sum((nxyz - xyz_plus) ** 2, dim=3, keepdim=False)) # B * N * N
    k_idx = torch.argsort(dist, dim=2, descending=False)[:, :, 1:K + 1] # B * N * K [:, :, 0]是原点舍弃掉
    indice = torch.arange(B).view(B, 1, 1).repeat(1, N, K)
    k_xyz = xyz[indice, k_idx, :] # B * N * K * 3
    k_dist = k_xyz - xyz.view(B, N, 1, 3).repeat(1, 1, K, 1) # B * N * K * 3
    return k_idx, k_dist

def FPS(xyz, dim_out):
    """
    xyz: B * N * 3
    return:
    idx: B * dim_out
    """
    B, N, _ = xyz.shape
    O = dim_out
    idx = torch.zeros((B, O), dtype=int)
    now = torch.randint(0, N, (B, 1))
    dist = torch.ones((B, N)) * 1e10 
    indice = torch.arange(B).view(B, 1)
    for i in range(O):
        idx[:, i] = now
        xyz_now = xyz[indice, now, :].repeat(1, N, 1) # B * 1 * 3 -> B * N * 3
        now_dist = torch.sqrt(torch.sum((xyz_now - xyz)**2, dim=2, keepdim=False)) # B * N
        mask = now_dist < dist
        dist[mask] = now_dist[mask]
        now = torch.argmax(dist, dim=1,keepdim=True)
    return idx