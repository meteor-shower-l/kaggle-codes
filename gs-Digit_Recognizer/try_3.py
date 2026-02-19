# 使用LeNet

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


if __name__ == "__main__":
    # 指定设备为gpu
    device = torch.device("cuda")
    # 指定学习率
    lr = 1e-3
    # 指定训练轮数
    epoch = 200
    #
    weiter = SummaryWriter(log_dir="log")

    # 读取数据，并创建dataset、dataloader
    init_features, init_lables = read_csv(
        "F:\\code_files\\python\\kaggle\\gs-Digit_Recognizer\\data\\train.csv"
    )
    # 将向量reshape为正常的28*28
    init_features = torch.reshape(init_features, (-1, 1, 28, 28))
    init_dataset = TensorDataset(init_features, init_lables)
    # 将初始数据分割为训练集与测试集
    total_size = len(init_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(
        init_dataset, [train_size, test_size]
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )

    # 定义网络
    net = nn.Sequential(
        nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1
        ),  # (6,28,28)
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),  # (6,14,14)
        nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1
        ),  # (16,10,10)
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),  # (16,5,5)
        nn.Flatten(),
        nn.Linear(in_features=16 * 5 * 5, out_features=120),
        nn.Sigmoid(),
        nn.Linear(in_features=120, out_features=84),
        nn.Sigmoid(),
        nn.Linear(in_features=84, out_features=10),
    )

    net = net.to(device)
    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=lr,
    )
    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",  # 特征越大越好
        factor=0.8,  # 每次衰减的系数
        patience=10,  # 若10轮内，参数没有突破原有最大/最小值，学习率就会衰减
        cooldown=5,  # 衰减后5轮内不会再次检测
        threshold=0.0001,  # 只有超过0.0001的变化才会被视为改善
        min_lr=1e-5,  # 学习率不小于1e-5
    )

    for train_times in range(epoch):
        net.train()
        total_loss = 0
        total_times = 0
        for features, lables in train_dataloader:
            features = features.to(device)
            lables = lables.to(device)
            output = net(features)
            loss = loss_function(output, lables)
            total_loss += loss
            total_times += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        average_loss = total_loss / total_times
        net.eval()
        correct = 0
        total = 0
        for features, lables in test_dataloader:
            features = features.to(device)
            lables = lables.to(device)
            output = net(features)
            _, predicted = torch.max(output.data, 1)
            total += lables.size(0)
            correct += (predicted == lables).sum().item()
        accuracy = correct / total
        scheduler.step(accuracy)
        current_lr = scheduler.get_last_lr()[0]
        print(f"""第{train_times+1}轮训练后
平均损失为{average_loss}
在测试集上的正确率为{accuracy*100}%
当前学习率: {current_lr}""")
        weiter.add_scalar("平均损失", average_loss, train_times)
        weiter.add_scalar("平均正确率", accuracy, train_times)
