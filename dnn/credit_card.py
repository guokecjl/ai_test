# coding: utf-8

'''
通过dnn找出有可能流失的信用卡用户
数据集:https://www.kaggle.com/sakshigoyal7/credit-card-customers
CLIENTNUM: 用户ID
Attrition_Flag: 账户标识，Existing Customer表示账户未关闭，Attrited Customer表示账户已关闭
Customer_Age：账户年龄
Gender: 性别男M女F
Dependent_count: 副卡数量
Education_Level:　教育水平
Marital_Status:  婚姻状况
Income_Category: 收入类别
Card_Category: 卡类型
Months_on_book: 使用时间(月)
Total_Relationship_Count：　产品数量
Months_Inactive_12_mon: 过去12个月未使用的月份数
Contacts_Count_12_mon：　最近12个月联系人数量
Credit_Limit: 信用卡额度
Total_Revolving_Bal: 信用卡余额
Avg_Open_To_Buy：　过去１２个月平均信用额度
Total_Amt_Chng_Q4_Q1：　q4季度的交易金额/q1季度的交易金额
Total_Trans_Amt：　总交易金额
Total_Trans_Ct：　总交易数量
Total_Ct_Chng_Q4_Q1： 交易数量变更
Avg_Utilization_Ratio：平局卡利用率
'''
import os
import sys
sys.path.append(os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".."))
base_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from tools.data_analysis import show_miss_value, caculate_corr

data_set = pd.read_csv(os.path.join(base_dir, 'BankChurners.csv'))

# 分析数据
# show_miss_value(data_set)
# caculate_corr(data_set)

# 去掉无用特征
data_set.drop('CLIENTNUM', axis=1, inplace=True)
# columns = ["Customer_Age", 'Dependent_count', 'Months_on_book',
#        'Total_Relationship_Count', 'Months_Inactive_12_mon',
#        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
#        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
#        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
# for column_name in columns:
#     Q1 = data_set[column_name].quantile(0.25)
#     Q3 = data_set[column_name].quantile(0.75)
#     IQR = Q3 - Q1
#     data_set = data_set[~((data_set[column_name] < (Q1 - 3 * IQR)) |(data_set[column_name] > (Q3 + 3 * IQR)))]
data_set.drop('Credit_Limit', axis=1, inplace=True)
# 将正负样本转换为0,1
data_set['Attrition_Flag'] = data_set['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})
# one_hot转换
data_dummies = pd.get_dummies(data_set)
y = data_dummies.pop('Attrition_Flag').to_numpy()
x = torch.tensor(data_dummies.to_numpy())
x = (x.float() - x.min(axis=0)[0])/(x.max(axis=0)[0] - x.min(axis=0)[0])
x = x.numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y_train = y_train.reshape(-1, 1)
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)
train_loader = torch.utils.data.DataLoader(np.concatenate((x_train, y_train), 1), batch_size=64, drop_last=True, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(36, 10),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(10, 2),
            # nn.ReLU()
        )

    def forward(self, x):
        x = x.reshape(-1, 36)
        x = self.model(x)
        return x


def train_model(num):
    net = Net()
    optim = torch.optim.Adam(params=net.parameters())
    loss_func = nn.CrossEntropyLoss()
    net.train()
    for epoch in range(num):
        total_loss = 0
        for data in train_loader:
            input, label = data[:, :-1].float(), data[:, -1].long()
            output = net(input)
            loss = loss_func(output, label)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(total_loss)

    net.eval()
    eval_y = net(x_test.float())
    _, eval_y = eval_y.max(axis=1)
    print((eval_y == y_test).sum().item()/len(y_test))
    print(float(eval_y[y_test == 1].sum().item())/y_test.sum().item())
    conf_mtrx(y_test, eval_y)

def conf_mtrx(y_test, y_pred):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    f, ax = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f",
                ax=ax)
    plt.xlabel("predicted y values")
    plt.ylabel("real y values")
    plt.title("\nConfusion Matrix")
    plt.show()

train_model(100)
