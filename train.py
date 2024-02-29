import torch
from torch.utils.data import DataLoader
from pytorch3d.datasets import ModelNet
import torch.optim as optim
from utils.data_loader import *
import argparse
from model.PointTransformer import *


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_train = ModelNet(root='./dataset', split='train', download='True')


dataset_size = len(dataset_train)
num_train = 0.8 * dataset_size
num_val = dataset_size - num_train
loader_train = DataLoader(dataset_train, batch_size=32, sampler= ChunkSampler(num_train, 0))
loader_val = DataLoader(dataset_train, batch_size=32, sampler= ChunkSampler(num_val, num_train))

model = PointTransformer(k_nums=16, class_num=40)
loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

train_loss = []
valid_loss = []
min_loss = 1000
for epoch in args.epochs:
    # train
    model.train()
    train_epoch_loss = []
    idx = 0
    for data_x, data_y in loader_train:
        data_x = data_x.to(torch.float32).to(args.devide)
        data_y = data_y.to(torch.float32).to(args.devide).long()
        output = model(data_x)
        optimizer.zero_grad()
        loss = loss_f(output, data_y)
        train_epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        if idx % (len(loader_train) // 2):
            print(f"epoch = {epoch}/{args.epochs}, {idx}/{len(loader_train)}of train, loss = {loss.item()}")
        idx += 1
    train_loss.append(sum(train_epoch_loss) / len(train_epoch_loss))

    # valid
    model.eval()
    valid_epoch_loss = []
    total = 0
    right = 0
    for data_x, data_y in loader_val:
        data_x = data_x.to(torch.float32).to(device)
        data_y = data_y.to(torch.float32).to(device).long()
        output = model(data_x)
        optimizer.zero_grad()
        loss = loss_f(output, data_y)
        valid_epoch_loss.append(loss.item())
        _, pred = torch.argmax(output, dim=1)
        right += (pred == data_y).sum()
        total += pred.shape[0]
    valid_loss.append(sum(valid_epoch_loss) / len(valid_epoch_loss))
    print(f"Accuracy: {100 * right / total}%")
    print(f"Train loss: {train_loss[-1]}")
    print(f"Valid loss: {valid_loss[-1]}")

    if valid_loss[-1] < min_loss:
        print(f"Valid loss decrease({min_loss}->{valid_loss[-1]}). Saving model...")
        torch.save(model.state_dict(), 'checkpoint/PointTransformer.pt')
        min_loss = valid_loss[-1]
