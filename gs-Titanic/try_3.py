# 尝试使用集成学习中的Stacking方法
# 经查阅资料，Stacking方法要求使用k折交叉验证
# 但是本处决定尝试随机采样
# 基学习器使用前2次的MLP
# 训练5个基学习器,使用线性模型集成基学习器结果
# 数据使用train_processed_2.csv
# 数据中20%用于训练线性模型，80%用于Stacking法训练基学习器

# 效果得到一定提升(0.78468)，可以考虑采用正常的k折交叉验证并更改基学习器架构以确保好而不同
import pandas as pd
import torch
from torch import nn
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split,
    SubsetRandomSampler,
)
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda")


def read_csv(file_dir):
    df = pd.read_csv(file_dir)
    X = df.iloc[:, 1:].values  # 特征
    Y = df.iloc[:, 0].values  # 标签
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y


# 定义网络生成函数
def generate_net():
    net = nn.Sequential(
        nn.Linear(34, 64),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, 2),
    )
    net = net.to(device)
    return net


# 定义损失函数为交叉熵损失函数
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)


# 定义优化器生成函数
def generate_optimizer(given_net, given_lr=1e-4):
    optimizer = torch.optim.Adam(given_net.parameters(), lr=given_lr)
    return optimizer


# 读取初始数据
init_data_dir = (
    "F:\\code_files\\python\\kaggle\\gs-Titanic\\data\\train_processed_2.csv"
)
init_X, init_Y = read_csv(init_data_dir)
init_X = init_X.to(device)
init_Y = init_Y.to(device)
init_dataset = TensorDataset(init_X, init_Y)
"""
# 将初始数据分割为训练集与测试集
# 在最终预测时，不再需要测试集
total_size = len(init_dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size
train_dataset, test_dataset = random_split(
    init_dataset, [train_size, test_size]
)
"""
# 将训练集分割为基学习器集与元学习器集
base_size = int(0.8 * len(init_dataset))
meta_size = len(init_dataset) - base_size
base_dataset, meta_dataset = random_split(init_dataset, [base_size, meta_size])
# 使用随机采样法得到5个基训练集
base_indices = list(range(len(base_dataset)))
bagging_subsets = []
for i in range(5):
    # 有放回地随机采样生成新索引
    bootstrap_indices = torch.randint(
        0, len(base_dataset), (len(base_dataset),)
    ).tolist()
    sampler = SubsetRandomSampler(bootstrap_indices)
    bagging_subsets.append(sampler)  # 每个sampler对应一个基学习器的训练数据
base_loaders = []
# 创建基训练数据加载器
for sampler in bagging_subsets:
    loader = DataLoader(base_dataset, batch_size=32, sampler=sampler)
    base_loaders.append(loader)
# 元数据加载器
meta_loader = DataLoader(
    meta_dataset, batch_size=32, shuffle=False, drop_last=False
)

# 读取预测数据
predict_data_dir = (
    "F:\\code_files\\python\\kaggle\\gs-Titanic\\data\\test_processed_2.csv"
)
predict_X, predict_Y = read_csv(predict_data_dir)
predict_X = predict_X.to(device)
predict_Y = predict_Y.to(device)
# 定义基训练轮数
base_epoch = 500
# 定义5个独立的网络(基学习器)
base_net_list = []
for i in range(5):
    net_generated = generate_net()
    base_net_list.append(net_generated)

# 分别训练5个独立的网络(基学习器)
for index in range(5):
    print(f"开始第{index+1}个基学习器的训练")
    work_net = base_net_list[index]
    work_dataloader = base_loaders[index]
    optimizer = generate_optimizer(work_net)
    for i in range(base_epoch):
        work_net.train()
        total_loss = 0
        total_times = 0
        for features, label in work_dataloader:
            predict = work_net(features)
            loss = loss_function(predict, label)
            total_loss += loss.item()
            total_times += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"完成第{i+1}轮训练,损失:{total_loss / total_times}")

# 定义元学习器
meta_net = nn.Sequential(nn.Linear(10, 2)).to(device)
# 定义元学习器优化器
meta_optimizer = torch.optim.Adam(params=meta_net.parameters())
# 定义元训练轮数
meta_epoch = 250


# 生成元特征
def generate_meta_data(f_net_list, f_meta_loader):
    meta_features = []
    meta_labels = []
    # 设置为预测模式
    for net in f_net_list:
        net.eval()
    with torch.no_grad():  # 不计算梯度
        for data, lable in f_meta_loader:
            batch_meta_features = []
            for net in f_net_list:
                raw_output = net(data)
                # 使用softmax得到概率，以对齐不同模型输出尺度的不同
                # 从而降低元学习器的学习难度
                probability = torch.softmax(raw_output, dim=1)
                batch_meta_features.append(probability)
            # 张量形状变为(batch_size,10)
            stacked_features = torch.cat(batch_meta_features, dim=1)
            meta_features.append(stacked_features)
            meta_labels.append(lable)
        # 融合各个batch的张量并返回,形状为(total_num,10),(total_num,1)
        return torch.cat(meta_features), torch.cat(meta_labels)


# 定义元学习器预测函数
# 之所以在此处定义，是因为将在训练函数中调用，用于估计元学习器的训练结果与拟合状况
def meta_net_predict(f_base_nets, f_meta_net, X):
    base_predictions = []
    for net in f_base_nets:
        net.eval()
        with torch.no_grad():
            raw_output = net(X)
            prob_output = torch.softmax(raw_output, dim=1)
            base_predictions.append(prob_output)
    meta_input = torch.cat(base_predictions, dim=1)
    f_meta_net.eval()
    with torch.no_grad():
        final_output = meta_net(meta_input)
        _, prediction = torch.max(final_output, 1)
    return prediction


meta_features, meta_labels = generate_meta_data(base_net_list, meta_loader)
final_meta_dataset = TensorDataset(meta_features, meta_labels)
final_meta_dataloader = DataLoader(
    dataset=final_meta_dataset, batch_size=32, shuffle=True
)
writer = SummaryWriter(log_dir="log")
# 训练元学习器
print("开始训练元学习器")
for epoch in range(meta_epoch):
    meta_net.train()
    total_loss = 0
    total_times = 0
    for feature, label in final_meta_dataloader:
        output = meta_net(feature)
        loss = loss_function(output, label)
        total_loss += loss.item()
        total_times += 1
        meta_optimizer.zero_grad()
        loss.backward()
        meta_optimizer.step()
    print(f"第{epoch}轮训练完成,loss:{total_loss/total_times}")
# 产生最终预测结果并保存
final_result = meta_net_predict(base_net_list, meta_net, predict_X)
final_result_cpu = final_result.cpu().numpy()
results_df = pd.DataFrame(
    {
        "PassengerId": range(892, 892 + len(final_result)),
        "Survived": final_result_cpu,
    }
)
results_df.to_csv("submission_3.csv", index=False)
