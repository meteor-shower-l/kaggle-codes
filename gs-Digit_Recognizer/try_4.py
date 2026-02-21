# 使用集成学习方法，综合前3种尝试
# 数据集划分为0.64基学习器，0.16元学习器，0.1验证集，0.1测试集

# 经过实验，发现在当前参数下，以64%数据训基学习器可以在验证集上达到较好的效果
# 但是元学习器最终在测试集上的表现不如基学习器的表现，原因未知
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from try_1 import mordern_cnn_net
from try_2 import LVQ_DIY
from try_3 import LeNet


def read_csv_train(file_dir):
    df = pd.read_csv(file_dir)
    X = df.iloc[:, 1:].values  # 特征
    Y = df.iloc[:, 0].values  # 标签
    return X, Y


def read_csv_predict(file_dir):
    df = pd.read_csv(file_dir)
    X = df.iloc[:, :].values  # 特征
    return X


def train(f_net_list, f_LVQ, f_lr, f_epoch):
    # 定义损失函数
    loss_function = nn.CrossEntropyLoss().to(device)
    # 训练神经网络
    for i in [0, 1]:
        print(f"正在训练第{i+1}个神经网络")
        work_net = f_net_list[i]
        # 定义优化器
        optimizer = torch.optim.Adam(work_net.parameters(), lr=f_lr)
        # 定义学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            patience=10,
            threshold=0.0001,
            min_lr=1e-5,
            factor=0.8,
        )
        # 训练
        for times in range(f_epoch):
            total_loss = 0
            total_times = 0
            for features, lables in base_train_dataloader:
                # 训练
                work_net.train()
                features = features.to(device)
                lables = lables.to(device)
                output = work_net(features)
                loss = loss_function(output, lables)
                total_loss += loss.item()
                total_times += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            average_loss = total_loss / total_times
            # 测试并调整学习率
            work_net.eval()
            with torch.no_grad():
                output = work_net(
                    square_tensor_verification_features.to(device)
                )
                _, predict = torch.max(output.data, dim=1)
            correct = (
                (predict == tensor_verification_lables.to(device)).sum().item()
            )
            total = len(tensor_verification_lables)
            accuracy = correct / total
            scheduler.step(accuracy)
            current_lr = scheduler.get_last_lr()[0]
            print(f"""第{times+1}轮训练后
    平均损失为{average_loss}
    在验证集上的正确率为{accuracy*100}%
    当前学习率: {current_lr}""")
            weiter.add_scalar(f"学习器{i}平均损失", average_loss, times)
            weiter.add_scalar(f"学习器{i}平均正确率", accuracy, times)
    # 训练LVQ
    print("正在训练第3个学习器")
    f_LVQ.fit(LVQ_base_features, LVQ_base_lables)


if __name__ == "__main__":
    # 定义超参数
    # 指定设备为gpu
    device = torch.device("cuda")
    # 指定基学习率
    base_lr = 1e-3
    # 指定元学习率
    meta_lr = 1e-3
    # 指定基训练轮数
    base_epoch = 200
    # 指定元训练轮数
    meta_epoch = 300
    # 指定PCA维度
    dim_PCA = 100
    weiter = SummaryWriter(log_dir="log")

    # 定义基学习器列表
    mordern_cnn = mordern_cnn_net(device)
    LeNet_cnn = LeNet(device)
    LVQ = LVQ_DIY(
        f_distance_dim=2,
        f_num_class=10,
        f_num_prototypes=3,
        f_epoch=base_epoch,
        f_lr=base_lr,
        f_decay_rate=0.99,
        f_device=device,
    )
    base_net_list = [mordern_cnn, LeNet_cnn, LVQ]

    # 定义元学习器
    meta_net = nn.Linear(30, 10)

    # 读取数据集(均为array)
    init_features, init_lables = read_csv_train(
        "F:\\code_files\\python\\kaggle\\gs-Digit_Recognizer\\data\\train.csv"
    )
    predict_features = read_csv_predict(
        "F:\\code_files\\python\\kaggle\\gs-Digit_Recognizer\\data\\test.csv"
    )
    # 划分为基训练集、元训练集、测试集
    # 将数据集划分为训练集和测试验证集
    (
        train_features,
        test_and_verification_features,
        train_lables,
        test_and_verification_lables,
    ) = train_test_split(
        init_features, init_lables, test_size=0.2, random_state=42
    )
    # 将测试验证集划分为测试集和验证集
    test_features, verification_features, test_lables, verification_lables = (
        train_test_split(
            test_and_verification_features,
            test_and_verification_lables,
            test_size=0.5,
            random_state=42,
        )
    )
    # 将训练集划分为基训练集和元训练集
    base_features, meta_features, base_lables, meta_lables = train_test_split(
        train_features, train_lables, test_size=0.2, random_state=42
    )

    # 转换为Tensor
    tensor_base_features = torch.tensor(base_features, dtype=torch.float32)
    tensor_meta_features = torch.tensor(meta_features, dtype=torch.float32)
    tensor_verification_features = torch.tensor(
        verification_features, dtype=torch.float32
    )
    tensor_test_features = torch.tensor(test_features, dtype=torch.float32)
    tensor_predict_features = torch.tensor(
        predict_features, dtype=torch.float32
    )

    tensor_base_lables = torch.tensor(base_lables, dtype=torch.long)
    tensor_meta_lables = torch.tensor(meta_lables, dtype=torch.long)
    tensor_verification_lables = torch.tensor(
        verification_lables, dtype=torch.long
    )
    tensor_test_lables = torch.tensor(test_lables, dtype=torch.long)

    # 为神经网络准备数据
    # 基训练集
    square_tensor_base_features = torch.reshape(
        tensor_base_features, (-1, 1, 28, 28)
    )
    # 元训练集
    square_tensor_meta_features = torch.reshape(
        tensor_meta_features, (-1, 1, 28, 28)
    )
    # 验证集
    square_tensor_verification_features = torch.reshape(
        tensor_verification_features, (-1, 1, 28, 28)
    )
    # 测试集
    square_tensor_test_features = torch.reshape(
        tensor_test_features, (-1, 1, 28, 28)
    )

    # 定义Datast与DataLoader
    base_train_dataset = TensorDataset(
        square_tensor_base_features, tensor_base_lables
    )
    base_train_dataloader = DataLoader(
        base_train_dataset,
        batch_size=256,
        shuffle=True,
        persistent_workers=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    # 为LVQ准备数据
    pca = PCA(n_components=dim_PCA)
    # 用于基训练的数据
    LVQ_base_features = pca.fit_transform(base_features)
    LVQ_base_features = torch.from_numpy(LVQ_base_features).float().to(device)
    LVQ_base_lables = torch.from_numpy(base_lables).long().to(device)
    # 用于元训练的数据
    LVQ_meta_features = pca.transform(meta_features)
    LVQ_meta_features = torch.tensor(LVQ_meta_features, dtype=torch.float32)
    # 验证数据
    LVQ_verification_features = pca.transform(verification_features)
    LVQ_verification_features = torch.tensor(
        LVQ_verification_features, dtype=torch.float32
    )
    # 用于测试的数据
    LVQ_test_features = pca.transform(test_features)
    LVQ_test_features = torch.tensor(LVQ_test_features, dtype=torch.float32)
    # 预测数据
    LVQ_predict_features = pca.transform(predict_features)
    LVQ_predict_features = torch.tensor(
        LVQ_predict_features, dtype=torch.float32
    )

    # 训练基学习器
    train(
        f_net_list=base_net_list, f_LVQ=LVQ, f_lr=base_lr, f_epoch=base_epoch
    )

    # 准备元训练数据(分离梯度并移动至cpu)
    meta_features_1 = (
        torch.nn.functional.softmax(
            base_net_list[0](square_tensor_meta_features.to(device)), dim=1
        )
        .detach()
        .cpu()
    )
    meta_features_2 = (
        torch.nn.functional.softmax(
            base_net_list[1](square_tensor_meta_features.to(device)), dim=1
        )
        .detach()
        .cpu()
    )
    meta_features_3 = (
        (base_net_list[2].predict_proba(LVQ_meta_features.to(device)))
        .detach()
        .cpu()
    )
    final_meta_features = torch.cat(
        [meta_features_1, meta_features_2, meta_features_3], dim=1
    )
    final_meta_lables = tensor_meta_lables.cpu()
    meta_Dataset = TensorDataset(final_meta_features, final_meta_lables)
    meta_DataLoader = DataLoader(
        meta_Dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )
    # 准备验证数据
    processed_verification_features_1 = (
        torch.nn.functional.softmax(
            base_net_list[0](square_tensor_verification_features.to(device)),
            dim=1,
        )
        .detach()
        .cpu()
    )
    processed_verification_features_2 = (
        torch.nn.functional.softmax(
            base_net_list[1](square_tensor_verification_features.to(device)),
            dim=1,
        )
        .detach()
        .cpu()
    )
    processed_verification_features_3 = (
        base_net_list[2]
        .predict_proba(LVQ_verification_features.to(device))
        .detach()
        .cpu()
    )
    final_verification_features = torch.cat(
        [
            processed_verification_features_1,
            processed_verification_features_2,
            processed_verification_features_3,
        ],
        dim=1,
    )
    final_verification_lables = tensor_verification_lables
    # 定义损失函数
    loss_function = nn.CrossEntropyLoss().to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(meta_net.parameters(), lr=meta_lr)
    # 定义学习器调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        patience=10,
        threshold=0.0001,
        min_lr=1e-5,
        factor=0.8,
    )

    # 训练元学习器
    for times in range(meta_epoch):
        total_loss = 0
        total_times = 0
        meta_net.train()
        for features, lables in meta_DataLoader:
            output = meta_net(features)
            loss = loss_function(output, lables)
            total_loss += loss
            total_times += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        average_loss = total_loss / total_times
        meta_net.eval()
        with torch.no_grad():
            output = meta_net(final_verification_features)
            j_, predict = torch.max(output.data, dim=1)
            correct = (predict == final_verification_lables).sum().item()
            total = len(final_verification_lables)
            accuracy = correct / total
        scheduler.step(accuracy)
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"完成元学习器第{times+1}轮训练，在元学习集上平均损失为{average_loss},在验证集上正确率为{accuracy}"
        )
        weiter.add_scalar("元学习器平均损失", average_loss, times)
        weiter.add_scalar("验证集正确率", accuracy, times)
    print("完成元学习器的训练")

    # 准备测试数据
    processed_test_features_1 = (
        torch.nn.functional.softmax(
            base_net_list[0](square_tensor_test_features.to(device)), dim=1
        )
        .detach()
        .cpu()
    )
    processed_test_features_2 = (
        torch.nn.functional.softmax(
            base_net_list[1](square_tensor_test_features.to(device)), dim=1
        )
        .detach()
        .cpu()
    )
    processed_test_features_3 = (
        base_net_list[2]
        .predict_proba(LVQ_test_features.to(device))
        .detach()
        .cpu()
    )
    final_test_features = torch.cat(
        [
            processed_test_features_1,
            processed_test_features_2,
            processed_test_features_3,
        ],
        dim=1,
    )
    final_test_lables = tensor_test_lables
    # 对测试数据进行分类
    test_output = meta_net(final_test_features)
    _, predict = torch.max(test_output.data, dim=1)
    correct = (predict == final_test_lables).sum().item()
    total = len(final_test_lables)
    accuracy = correct / total
    print(f"在测试数据集上的正确率为{accuracy}")
