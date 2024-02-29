import torch
import torch.nn as nn
from sample import *

class PTBlock(nn.module):
    def __init__(self, dim_features, dim_model_features, k_num):
        super().__init__()
        self.k_num = k_num
        self.fc1 = nn.Linear(dim_features, dim_model_features)
        self.fc2 = nn.Linear(dim_model_features, dim_features)
        self.Wk = nn.Linear(dim_model_features, dim_model_features, bias=False)
        self.Wq = nn.Linear(dim_model_features, dim_model_features, bias=False)
        self.Wv = nn.Linear(dim_model_features, dim_model_features, bias=False)
        self.pos_embd = nn.Sequential(
            nn.Linear(3, dim_model_features),
            nn.ReLU(),
            nn.Linear(dim_model_features, dim_model_features)
        )
        self.gamma = nn.Sequential(
            nn.Linear(dim_model_features, dim_model_features),
            nn.ReLU(),
            nn.Linear(dim_model_features, dim_model_features)
        )
    def forward(self, xyz, x):
        # x : B * N * D
        # xyz : B * N * 3
        B, N, F = x.shape
        K = self.k_num
        #计算k个索引
        k_idx, k_dist = compute_kidx(xyz, K) # B * N * K, B * N * K * 3
        res = x
        x = self.fc1(x) # B * N * F
        q = self.Wq(x).views(B, N, 1, F) # B * N * 1 * F
        k = sample_k(self.Wk(x), k_idx) # B * N * F -> B * N * K * F
        v = sample_k(self.Wv(x), k_idx) # B * N * F -> B * N * K * F
        pos = self.pos_embd(xyz.view(B, N, 1, 3).repeat(1, 1, K, 1) - k_dist) # B * N * K * F
        x = self.gamma(q.repeat(1, 1, K, F) - k + pos) / torch.sqrt(F) # B * N * K * F
        x = nn.Softmax(x, dim=3) 
        x = torch.einsum("bnkf,bnkf->bnf", x, v + pos) # B * N * F
        x = self.fc2(x) # B * N * D
        x = x + res
        return xyz, x
        
class TDBlock(nn.module):
    def __init__(self, down_rate, features_in, features_out, k_num):
        super().__init__()
        self.k_num = k_num
        self.down_rate = down_rate
        self.fcn = nn.Sequential(
            nn.Linear(features_in, features_out),
            nn.BatchNorm1d(features_out),
            nn.ReLU()
        )

    def forward(self, xyz, x):
        # x: B * N * D
        # xyz : B * N * 3
        B, N, F = x.shape
        O = N / self.down_rate
        K = self.k_num

        # FCN
        x = self.fcn(x)

        # FPS and KNN
        fps_idx = FPS(xyz, O) # B * O
        k_idx, _ = compute_kidx(xyz, K) # B * N * K
        indice = torch.arange(B).view(B, 1).repeat(1, O) # B * O
        final_idx = k_idx[indice, fps_idx, :] # B * O * K
        xyz = xyz[indice, fps_idx, :]

        # MaxPool
        indice = torch.arange(B).view(B, 1, 1).repeat(1, O, K) # B * O * K
        x = x[indice, final_idx, :] # B * O * K * F
        x = torch.max(x, dim=2, keepdim=False) # B * O * F
        
        return xyz, x
        