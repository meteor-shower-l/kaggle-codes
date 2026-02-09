# 目前进行5折交叉验证，正确率在75%-85%波动
# 且在保持MLP架构的基础上、增加隐藏层参数数未能提高正确率
# 尚不清楚正确率不高的原因
# 可能的一个原因是数据预处理时去除了若干有价值的信息

# 针对目前的参数，模型可以收敛且未观察到过拟合现象
# 采用多层感知机、ReLU激活函数、softmax输出层

import pandas as pd
import torch
from torch import nn
from k_fold_cross_validation import k_cross


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
    nn.Linear(64, 2),
)
net = net.to(device)
# 定义损失函数为交叉熵损失函数
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
# 定义优化器
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
# 读取训练数据
train_data_dir = (
    "F:\\code_files\\python\\kaggle\\gs-Titanic\\data\\train_processed.csv"
)
train_X, train_Y = read_csv(train_data_dir)
train_X = train_X.to(device)
train_Y = train_Y.to(device)
train_dataset = torch.utils.data.TensorDataset(train_X, train_Y)
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
)
# 读取预测数据
predict_data_dir = (
    "F:\\code_files\\python\\kaggle\\gs-Titanic\\data\\test_processed.csv"
)
predict_X, predict_Y = read_csv(predict_data_dir)
predict_X = predict_X.to(device)
predict_Y = predict_Y.to(device)
# 定义训练轮数
epoch = 500


# 定义初始化函数
def custom_init(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a=-0.1, b=0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


"""
k_cross(
    data_file=data_dir,
    net=net,
    initialization=custom_init,
    loss_function=loss_function,
    optimizer=optimizer,
    fold_num=5,
    epoch_num=epoch,
    logdir="log",
    rand_seed=36,
    device=device,
)
"""
# 训练
for i in range(epoch):
    for features, label in train_dataloader:
        net.train()
        predict = net(features)
        loss = loss_function(predict, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    net.eval()
    predict = net(train_X)
    loss = loss_function(predict, train_Y)
    _, predicted = torch.max(predict.data, 1)
    total = train_Y.size(0)
    correct = (predicted == train_Y).sum().item()
    batch_accuracy = 100.0 * correct / total
    print(f"损失:{loss},正确率:{batch_accuracy}")
    print(f"完成第{i+1}轮训练")
# 预测
net.eval()
with torch.no_grad():
    predict_result = net(predict_X)
    _, predicted_classes = torch.max(predict_result, 1)
    predicted_classes = predicted_classes.cpu().numpy()
    results_df = pd.DataFrame(
        {
            "PassengerId": range(892, 892 + len(predicted_classes)),
            "Survived": predicted_classes,
        }
    )
    results_df.to_csv("submission_1.csv", index=False)
