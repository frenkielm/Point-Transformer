import torch
import torch.nn as nn
from layers import *
class PointTransformer(nn.modules):
    def __init__(self, k_nums, class_num):
        super().__init__()
        self.K = k_nums
        self.C = class_num
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 32),
            nn.Relu(),
            nn.Linear(32, 32)
        )
        self.pointtransformer = nn.Sequential(
            PTBlock(32, 32, k_nums),
            TDBlock(4, 32, 64),
            PTBlock(64, 64, k_nums),
            TDBlock(4, 64, 128),
            PTBlock(128, 128, k_nums),
            TDBlock(4, 128, 256),
            PTBlock(256, 256, k_nums),
            TDBlock(4, 256, 512),
            PTBlock(512, 512, k_nums),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )

    def forward(self, xyz):
        x = self.mlp1(xyz)
        xyz, x = self.pointtransformer(xyz, x) # B * N * 512
        x = torch.mean(x, dim=1, keepdim=False) # B * 512
        x = self.mlp2(x) # B * C
        return x