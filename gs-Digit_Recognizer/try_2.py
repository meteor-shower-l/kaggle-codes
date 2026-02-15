# 使用学习向量法
# 考虑到LVQ的训练无需进行反向传播，故直接手动实现而不使用torch
# 考虑到后续可能需要将多种方法进行集成学习，故实现LVQ时输出为与每个原型向量的距离的相反数
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    ):
        self.distance_dim = f_distance_dim
        self.num_class = f_num_class
        self.num_prototypes = f_num_prototypes
        self.epoch = f_epoch
        self.lr = f_lr
        self.decay_rate = f_decay_rate

    # 训练函数
    def fit(self, X, Y):
        # 随机寻找指定类别的向量，作为原型向量
        n_sample, n_features = X.shape
        self.prototypes = np.zeros(
            (self.num_class, self.num_prototypes, n_features)
        )
        for i in range(self.num_class):
            mask = Y == i
            # 从X中选取mask=1的序号对应的值
            X_label = X[mask]
            random_choice = np.random.choice(
                len(X_label), size=self.num_prototypes, replace=False
            )
            selected_prototypes = X_label[random_choice]
            self.prototypes[i] = selected_prototypes
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
            current_lr = self.lr * np.power(self.decay_rate, train_times)
            # 将X与self.prototypes进行拓展以计算距离
            temp_X = X[
                :, np.newaxis, np.newaxis, :
            ]  # 形状为(n_sample, 1, 1, 100)
            temp_prototypes = self.prototypes[
                np.newaxis, :, :, :
            ]  # 形状为(1, num_class, num_prototypes, 100)
            diff = temp_X - temp_prototypes
            distance = np.power(
                np.sum(np.power(np.abs(diff), self.distance_dim), axis=-1),
                1 / self.distance_dim,
            )  # 形状为(n_sample,num_class,num_prototypes)
            flat_distance = distance.reshape(
                n_sample, -1
            )  # 压平为(n_sample,num_class*num_prototypes)
            # 获得最小距离的编号
            nearest_indices = np.argmin(flat_distance, axis=1)
            # 还原得到对应的类以及原型的编号
            nearest_class = nearest_indices // self.num_prototypes
            nearest_prototype = nearest_indices % self.num_prototypes
            # 获取增减系数，*2-1的目的是将0,1变为-1,1
            increase_decrease_coefficient = 2 * (nearest_class == Y) - 1
            # 获取每个样本对应的diff
            sample_indices = np.arange(
                n_sample
            )  # 创建一个(0,1,2...33599)的array
            # 抽出diff中的(i,nearest_class,nearest_prototype)的元素
            work_diff = diff[
                sample_indices, nearest_class, nearest_prototype, :
            ]  # 形状为(n_sample,n_features)
            # 更新数组为对应的diff*增减系数*学习率
            per_sample_update = (
                work_diff
                * increase_decrease_coefficient[:, np.newaxis]
                * current_lr
            )  # 形状为(n_sample,n_features)
            update_array = np.zeros_like(self.prototypes)
            # 将每个样本对应的改变量叠加至每个原型向量该变量
            np.add.at(
                update_array,
                (nearest_class, nearest_prototype),
                per_sample_update,
            )
            self.prototypes += update_array
            pred = self.predict(X)
            acc = np.mean(pred == Y)
            print(f"""Epoch {train_times+1},
lr: {current_lr},
train acc: {acc*100}%""")

    # 预测函数
    def predict(self, X):
        n_sample = X.shape[0]
        temp_X = X[:, np.newaxis, np.newaxis, :]  # 形状为(n_sample, 1, 1, 100)
        temp_prototypes = self.prototypes[
            np.newaxis, :, :, :
        ]  # 形状为(1, num_class, num_prototypes, 100)
        diff = temp_X - temp_prototypes
        distance = np.power(
            np.sum(np.power(np.abs(diff), self.distance_dim), axis=-1),
            1 / self.distance_dim,
        )  # 形状为(n_sample,num_class,num_prototypes)
        flat_distance = distance.reshape(
            n_sample, -1
        )  # 压平为(n_sample,num_class*num_prototypes)
        # 获得最小距离的编号
        nearest_indices = np.argmin(flat_distance, axis=1)
        nearest_class = nearest_indices // self.num_prototypes
        return nearest_class


# 指定学习率
lr = 1e-4
# 指定学习率衰减系数
decay_rate = 1
# 指定训练轮数
epoch = 200
# 指定PCA后维数
dim_PCA = 100

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

# 初始化LVQ分类器
lvq_machine = LVQ_DIY(
    f_distance_dim=2,
    f_num_class=10,
    f_num_prototypes=5,
    f_epoch=epoch,
    f_lr=lr,
    f_decay_rate=decay_rate,
)
# 训练
lvq_machine.fit(train_features, train_lables)
# 生成结果
test_result = lvq_machine.predict(test_features)
# 计算最终正确率
total_num = len(test_lables)
correct = np.sum((test_lables == test_result))
accuracy = correct / total_num
print(f"在测试集上,正确率为{accuracy*100}%")
