# 出现了未知的问题:loss与正确率始终不高
# 另外，即使是增加参数数，训练轮数，仍然无法在训练集上观察到过拟合现象
# 怀疑代码存在问题，考虑进一步检查代码并查阅相关文档
# 采用多层感知机、ReLU激活函数、softmax输出层

import pandas as pd
import torch
from torch import nn


def read_csv(file_dir):
    df = pd.read_csv(file_dir)
    X = df.iloc[:, 1:].values  # 特征
    Y = df.iloc[:, 0].values  # 标签
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y


device = torch.device("cuda")
# 定义网络
net = nn.Sequential(
    nn.Linear(29, 32),
    nn.ReLU(),
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 2),
)
net = net.to(device)
# 定义损失函数为交叉熵损失函数
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
# 定义优化器
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
# 读取数据
epoch = 100
X, Y = read_csv(
    "F:\\code_files\\python\\kaggle\\gs-Titanic\\data\\train_processed.csv"
)
X = X.to(device)
Y = Y.to(device)
train_dataset = torch.utils.data.TensorDataset(X, Y)
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
)
for i in range(epoch):
    for features, label in train_dataloader:
        net.train()
        predict = net(features)
        loss = loss_function(predict, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        net.eval()
        predict = net(features)
        loss = loss_function(predict, label)
        _, predicted = torch.max(predict.data, 1)
        total = label.size(0)
        correct = (predicted == label).sum().item()
        batch_accuracy = 100.0 * correct / total
        print(f"损失:{loss},正确率:{batch_accuracy}")
    print(f"完成第{i+1}轮训练")
