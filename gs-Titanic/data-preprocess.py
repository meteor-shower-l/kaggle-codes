import pandas
import csv
# 读取dataframe
def read_csv(file_dir):
    data = []
    with open(file_dir,'r',encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)
        for row in csv_reader:
            data.append(row)
    df = pandas.DataFrame(data, columns=headers)
    return df
# 将给定dataframe中的Name独热表示
def Name_onhot(df):
    return pandas.get_dummies(df,columns = ['Name'],dtype=int)
# 将给定dataframe中的Sex变为数字
def Sex_to_num(df):
    sex_mapping = {'male': 0, 'female': 1}
    df['Sex'] = df['Sex'].map(sex_mapping)
    return df
# 将给定dataframe中缺失的年龄替代为指定值
def insert_age(given_df,num):
    given_df['Age'] = given_df['Age'].fillna(num)
    return given_df
# 将给定dataframe中的ticket独热表示
def ticket_onehot(df):
    return pandas.get_dummies(df,columns = ['Ticket'],dtype=int)
# 将给定dataframe中的Cabin列填入Unknown后独热表示
def Cabin_onhot(df):
    for data in df['Cabin']:
        if data =='':
            data = 'Unknown'
    return pandas.get_dummies(df,columns = ['Cabin'],dtype=int)
# 将给定dataframe中的Embarked独热表示
def Embarked_onehot(df):
    return pandas.get_dummies(df,columns = ['Embarked'],dtype=int)

df1 = read_csv('F:\\code_files\\python\\kaggle\\gs-Titanic\\data\\train.csv')
df2 = read_csv('F:\\code_files\\python\\kaggle\\gs-Titanic\\data\\test.csv')
df = pandas.concat([df1,df2],ignore_index=True)

df = Name_onhot(df)
df = Sex_to_num(df)
df['Age'] = pandas.to_numeric(df['Age'], errors='coerce')
df = insert_age(df,int(df.Age.mean()))
df = ticket_onehot(df)
df = Cabin_onhot(df)
df = Embarked_onehot(df)
print(df)
split_index = 891
df_processed1 = (df[0:split_index]).iloc[:,1:]
df_processed2 = (df[split_index:]).iloc[:,1:]
df_processed1.to_csv('F:\\code_files\\python\\kaggle\\gs-Titanic\\data\\train_processed.csv')
df_processed2.to_csv('F:\\code_files\\python\\kaggle\\gs-Titanic\\data\\test_processed.csv')