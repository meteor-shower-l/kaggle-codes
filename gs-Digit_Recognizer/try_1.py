# 使用较浅的CNN
# 经实验，增加batchnorm层可以一定程度上提高正确率，但是原理未知
#
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter


def read_csv(file_dir):
    df = pd.read_csv(file_dir)
    X = df.iloc[:, 1:].values  # 特征
    Y = df.iloc[:, 0].values  # 标签
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y


# 指定设备为gpu
device = torch.device("cuda")
# 指定学习率
lr = 1e-3
# 指定训练轮数
epoch = 500
#
weiter = SummaryWriter(log_dir="log")
# 读取数据，并创建dataset、dataloader
init_features, init_lables = read_csv(
    "F:\\code_files\\python\\kaggle\\gs-Digit_Recognizer\\data\\train.csv"
)
# 将向量reshape为正常的28*28
init_features = torch.reshape(init_features, (-1, 1, 28, 28))
init_features = init_features.to(device)
init_lables = init_lables.to(device)
init_dataset = TensorDataset(init_features, init_lables)
# 将初始数据分割为训练集与测试集
total_size = len(init_dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size
train_dataset, test_dataset = random_split(
    init_dataset, [train_size, test_size]
)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=256, shuffle=True
)
test_dataloader = DataLoader(
    dataset=test_dataset, batch_size=256, shuffle=True
)

# 定义网络
net = nn.Sequential(
    nn.Conv2d(
        in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1
    ),  # 大小不变
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # 变为14*14
    nn.Conv2d(
        in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1
    ),  # 大小不变
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # 大小变为7*7
    nn.Flatten(),  # 展平为64*7*7的一维向量
    nn.Linear(64 * 7 * 7, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(512, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(128, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Linear(32, 10),
)
net = net.to(device)
# 定义损失函数
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
# 定义优化器
optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
for i in range(epoch):
    net.train()
    total_loss = 0
    total_times = 0
    for features, lables in train_dataloader:
        output = net(features)
        loss = loss_function(output, lables)
        total_loss += loss.item()
        total_times += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    average_loss = total_loss / total_times
    net.eval()
    correct = 0
    total = 0
    for features, lables in test_dataloader:
        output = net(features)
        _, predicted = torch.max(output.data, 1)
        total += lables.size(0)
        correct += (predicted == lables).sum().item()
    accuracy = correct / total
    print(
        f"第{i+1}轮训练后，平均损失为{average_loss},在测试集上的正确率为{accuracy*100}%"
    )
    weiter.add_scalar("平均损失", average_loss, i)
    weiter.add_scalar("平均正确率", accuracy, i)
