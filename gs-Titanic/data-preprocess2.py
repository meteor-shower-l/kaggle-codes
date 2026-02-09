# 去除Ticket、Cabin数据
# 对Name抽取头衔，并做向量化
# Sex、Embarked做向量化
# 增设家庭成员人数列
# 对年龄进行分组，并向量化

import pandas as pd
import csv


# 读取dataframe
def read_csv(file_dir):
    data = []
    with open(file_dir, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)
        for row in csv_reader:
            data.append(row)
    df = pd.DataFrame(data, columns=headers)
    return df


# 读取数据
df1 = read_csv("F:\\code_files\\python\\kaggle\\gs-Titanic\\data\\train.csv")
df2 = read_csv("F:\\code_files\\python\\kaggle\\gs-Titanic\\data\\test.csv")
df = pd.concat([df1, df2], ignore_index=True)

# 删除Cabin列和Ticket列
df.drop(["PassengerId", "Ticket", "Cabin"], axis=1, inplace=True)

# 从姓名列中提取头衔,并直接代替姓名列
df["Name"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
df["Name"] = df["Name"].str.strip()
df.rename(columns={"Name": "Title"}, inplace=True)

# 将空的年龄列替换为总数据（训练+测试）的年龄平均值
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
ave_age = round(df["Age"].mean(), 1)
df["Age"] = df["Age"].fillna(ave_age)
# 增加家庭成员数列
df["Family_member_num"] = pd.to_numeric(
    df["SibSp"], errors="coerce"
) + pd.to_numeric(df["Parch"], errors="coerce")
# 以3,18,60为分界线，对年龄进行划分
bins = [0, 3, 18, 60, 120]
labels = ["baby", "kid", "adult", "elder"]
df["Age sort"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

# 对头衔，性别，登船地点、年龄段进行独热表示
df = pd.get_dummies(
    df, columns=["Title", "Sex", "Embarked", "Age sort"], dtype=int
)

df_processed1 = df.iloc[:891]
df_processed2 = df.iloc[891:]
df_processed1.to_csv(
    "F:\\code_files\\python\\kaggle\\gs-Titanic\\data\\train_processed_2.csv",
    index=False,
)
df_processed2.to_csv(
    "F:\\code_files\\python\\kaggle\\gs-Titanic\\data\\test_processed_2.csv",
    index=False,
)
