import torch
from torch.utils.data import DataLoader
from pytorch3d.datasets import ModelNet
import torch.optim as optim
from utils.data_loader import *
import argparse
from model.PointTransformer import *

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
dataset_test = ModelNet(root='./dataset', split='test', download='True')
loader_test = DataLoader(dataset_test, batch_size=32)

right = 0
all = 0
model = PointTransformer()
model.load_state_dict(torch.load('checkpoint/PointTransformer.pt', map_location=device))
model.eval()
for data_x, data_y in loader_test:
    data_x = data_x.to(torch.float32).to(device)
    data_y = data_y.to(torch.float32).to(device).long()
    output = model(data_x)
    pred = torch.argmax(output, dim=1)
    right += (pred == data_y).sum()
    all += pred.shape[0]

print(f"Test Accuracy: {100 * right / all:.2f}%")