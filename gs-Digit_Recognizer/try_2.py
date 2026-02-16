# 使用学习向量法
# 考虑到LVQ的训练无需进行反向传播，故直接手动实现而不使用torch的Module
# 考虑到后续可能需要将多种方法进行集成学习，故实现LVQ时输出为与每个原型向量的距离的相反数

# 关于index_add_的说明：按照索引进行累积加法操作
# tensor.index_add_(dim, index, source)
# dim(int): 操作的维度；index(Tensor): 索引张量，指示在哪个位置添加；source(Tensor): 要添加的源张量

import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def read_csv_train(file_dir):
    df = pd.read_csv(file_dir)
    X = df.iloc[:, 1:].values  # 特征
    Y = df.iloc[:, 0].values  # 标签
    return X, Y


def read_csv_predict(file_dir):
    df = pd.read_csv(file_dir)
    X = df.iloc[:, :].values  # 特征
    return X


class LVQ_DIY:
    # 构造函数
    def __init__(
        self,
        f_distance_dim,
        f_num_class,
        f_num_prototypes,
        f_epoch,
        f_lr,
        f_decay_rate,
        f_device,
    ):
        self.distance_dim = f_distance_dim
        self.num_class = f_num_class
        self.num_prototypes = f_num_prototypes
        self.epoch = f_epoch
        self.lr = f_lr
        self.decay_rate = f_decay_rate
        self.device = f_device

    # 训练函数
    def fit(self, X, Y):
        # 不计算梯度以加快速度
        with torch.no_grad():
            # 随机寻找指定类别的向量，作为原型向量
            n_sample, n_features = X.shape
            self.prototypes = torch.zeros(
                (self.num_class, self.num_prototypes, n_features),
                device=self.device,
            )
            for i in range(self.num_class):
                mask = Y == i
                indices = torch.where(mask)[0]
                perm = torch.randperm(len(indices))
                selected_indices = indices[perm[: self.num_prototypes]]
                self.prototypes[i] = X[selected_indices]
            # 向量化计算更新向量，涉及多次维度转换
            # 首先将样本与原型进行广播并相减，得到diff(n_samples,num_class,num_prototype,n_features)
            # 计算距离(n_samples,num_class,num_prototype)
            # 将距离压缩为2维便于寻找最小值(n_samples,num_class*num_prototype)
            # 找到每个样本的最小值后再确定距离最近的class与prototype(n_sample,)(n_sample)
            # 比较每个样本的最近class与标签Y，确定增减系数(n_sample,)
            # 对每个样本，求出最近距离对应的diff，得到work_diff(n_sample,features)
            # work_diff*增减系数*学习率即为每个样本对应的更新向量(n_sample,features)
            # 结合确定的距离最近的class与prototype,对更新向量进行累加，得到最终的更新向量array(num_class,num_prototypes,n_features)
            # 每轮循环中将self.prototypes+=(得到的更新向量array)
            for train_times in range(self.epoch):
                # 令学习率随着轮数的增加而减少(指数级)
                current_lr = self.lr * (self.decay_rate**train_times)
                # 将X与self.prototypes进行拓展以计算距离
                temp_X = X[:, None, None, :]  # 形状为(n_sample, 1, 1, 100)
                temp_prototypes = self.prototypes[
                    None, :, :, :
                ]  # 形状为(1, num_class, num_prototypes, 100)
                diff = temp_X - temp_prototypes
                distance = torch.norm(
                    diff, p=self.distance_dim, dim=-1
                )  # 形状为(n_sample,num_class,num_prototypes)
                flat_distance = distance.reshape(
                    n_sample, -1
                )  # 压平为(n_sample,num_class*num_prototypes)
                # 获得最小距离的编号
                nearest_indices = torch.argmin(flat_distance, dim=1)
                # 还原得到对应的类以及原型的编号
                nearest_class = nearest_indices // self.num_prototypes
                nearest_prototype = nearest_indices % self.num_prototypes
                # 获取增减系数，*2-1的目的是将0,1变为-1,1
                increase_decrease_coefficient = 2 * (nearest_class == Y) - 1
                # 获取每个样本对应的diff
                sample_indices = torch.arange(
                    n_sample
                )  # 创建一个(0,1,2...33599)的array
                # 抽出diff中的(i,nearest_class,nearest_prototype)的元素
                work_diff = diff[
                    sample_indices, nearest_class, nearest_prototype, :
                ]  # 形状为(n_sample,n_features)
                # 更新数组为对应的diff*增减系数*学习率
                updates = (
                    work_diff
                    * increase_decrease_coefficient[:, None]
                    * current_lr
                )  # 形状为(n_sample,n_features)
                # 将每个样本对应的改变量叠加至每个原型向量该变量
                flat_indices = (
                    nearest_class * self.num_prototypes + nearest_prototype
                )  # 将索引压缩至一维，即每个原型只使用一个编号表示
                flat_updates = torch.zeros(
                    self.num_class * self.num_prototypes,
                    n_features,
                    device=self.device,
                )
                flat_updates.index_add_(0, flat_indices, updates)
                update_array = flat_updates.view(
                    self.num_class, self.num_prototypes, n_features
                )
                # 进行更新
                self.prototypes += update_array
                # 预测并计算正确率
                pred = self.predict_result(X)
                acc = (pred == Y).float().mean()
                print(f"""Epoch {train_times+1},
lr: {current_lr},
train acc: {acc*100}%""")

    # 预测函数
    def predict_result(self, X):
        # 不计算梯度以加快速度
        with torch.no_grad():
            n_sample = X.shape[0]
            temp_X = X[:, None, None, :]  # 形状为(n_sample, 1, 1, 100)
            temp_prototypes = self.prototypes[
                None, :, :, :
            ]  # 形状为(1, num_class, num_prototypes, 100)
            diff = temp_X - temp_prototypes
            distance = torch.norm(
                diff, p=self.distance_dim, dim=-1
            )  # 形状为(n_sample,num_class,num_prototypes)
            flat_distance = distance.reshape(
                n_sample, -1
            )  # 压平为(n_sample,num_class*num_prototypes)
            # 获得最小距离的编号
            nearest_indices = torch.argmin(flat_distance, dim=1)
            nearest_class = nearest_indices // self.num_prototypes
            return nearest_class

    def predict_distance(self, X):
        with torch.no_grad():
            temp_X = X[:, None, None, :]  # 形状为(n_sample, 1, 1, 100)
            temp_prototypes = self.prototypes[
                None, :, :, :
            ]  # 形状为(1, num_class, num_prototypes, 100)
            diff = temp_X - temp_prototypes
            distance = torch.norm(
                diff, p=self.distance_dim, dim=-1
            )  # 形状为(n_sample,num_class,num_prototypes)
            min_distance = -torch.min(distance, dim=-1)[
                0
            ]  # 形状为(n_sample,num_class)
            return min_distance


# 指定学习率
lr = 1e-4
# 指定学习率衰减系数
decay_rate = 1
# 指定训练轮数
epoch = 200
# 指定PCA后维数
dim_PCA = 100
# 指定设备
device = "cuda"

# 读取数据
init_features, init_lables = read_csv_train(
    "F:\\code_files\\python\\kaggle\\gs-Digit_Recognizer\\data\\train.csv"
)
predict_features = read_csv_predict(
    "F:\\code_files\\python\\kaggle\\gs-Digit_Recognizer\\data\\test.csv"
)
# 将数据集划分为训练集和测试集
train_features, test_features, train_lables, test_lables = train_test_split(
    init_features, init_lables, test_size=0.2, random_state=42
)
# 进行PCA降维以及标准化
# 其中在训练集上获得降维参数，施加在测试集和预测集上
pca = PCA(n_components=dim_PCA)
# (33600,100)
train_features = pca.fit_transform(train_features)
# (8400,100)
test_features = pca.transform(test_features)
predict_features = pca.transform(predict_features)
# 全部转化为tensor并移动到GPU
train_features = torch.from_numpy(train_features).float().to(device)
train_lables = torch.from_numpy(train_lables).long().to(device)
test_features = torch.from_numpy(test_features).float().to(device)
test_lables = torch.from_numpy(test_lables).long().to(device)
predict_features = torch.from_numpy(predict_features).float().to(device)


# 初始化LVQ分类器
lvq_machine = LVQ_DIY(
    f_distance_dim=2,
    f_num_class=10,
    f_num_prototypes=5,
    f_epoch=epoch,
    f_lr=lr,
    f_decay_rate=decay_rate,
    f_device=device,
)
# 训练
lvq_machine.fit(train_features, train_lables)
# 生成结果
test_result = lvq_machine.predict(test_features)
# 计算最终正确率
total_num = len(test_lables)
correct = (test_lables == test_result).float().sum()
accuracy = correct / total_num
print(f"在测试集上,正确率为{accuracy*100}%")
