# 实现k折交叉验证
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import pandas as pd


def read_csv(file_dir):
    df = pd.read_csv(file_dir)
    X = df.iloc[:, 1:].values  # 特征
    Y = df.iloc[:, 0].values  # 标签
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y


def k_cross(
    data_file,
    net,
    initialization,
    loss_function,
    optimizer,
    fold_num,
    epoch_num,
    logdir,
    rand_seed,
    device,
):
    # 创建log-writer,并将损失函数与模型转移至device上
    writer = SummaryWriter(log_dir=logdir)
    X, Y = read_csv(data_file)
    X = X.to(device)
    Y = Y.to(device)
    net.to(device)
    loss_function.to(device)
    # 生成k折划分
    kf = KFold(n_splits=fold_num, shuffle=True, random_state=rand_seed)
    # 划分为k折
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        # 划分数据并定义DataLoader
        train_dataset = TensorDataset(X[train_index], Y[train_index])
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=64, drop_last=True
        )
        # 每折都需要初始化模型
        net.apply(initialization)
        # 重置优化器状态
        optimizer = (
            type(optimizer)(net.parameters(), **optimizer.defaults)
            if hasattr(optimizer, "defaults")
            else type(optimizer)(net.parameters())
        )
        # 训练epoch_num轮
        for i in range(epoch_num):
            print(f"在{device}上开始第{i}轮训练与测试")
            net.train()
            total_loss = 0
            total_times = 0
            for batch_X, batch_Y in train_dataloader:
                batch_Y_hat = net(batch_X)
                loss = loss_function(batch_Y_hat, batch_Y)
                total_loss += loss.item()
                total_times += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            writer.add_scalar(
                f"train_loss,k-flod:{fold}", total_loss / total_times, i + 1
            )
            # 在完成一轮训练后，在测试集上进行测试，并写入在测试集的正确率
            net.eval()
            with torch.no_grad():
                predict = net(X[test_index])
                loss = loss_function(predict, Y[test_index])
                _, predicted = torch.max(predict.data, 1)
                total = Y[test_index].size(0)
                correct = (predicted == Y[test_index]).sum().item()
                accuracy = 100.0 * correct / total

            writer.add_scalar(f"accuracy,k_flod{fold}", accuracy, i + 1)
        if device != "cpu":
            torch.cuda.empty_cache()
