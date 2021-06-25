# coding: utf-8
'''
通过dnn识别手写数字集,自行构建训练集
'''
import os
base_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from sklearn.model_selection import train_test_split

data_set = pd.read_csv('/home/zhouxi/my_product/ai_test/cnn/mnist_csv.csv')
label = data_set.pop('0').to_numpy()
train_data = data_set.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(train_data, label, test_size=0.2)
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)
train_loader = torch.utils.data.DataLoader(np.concatenate((x_train, y_train.reshape(-1,1)), 1), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(np.concatenate((x_test, y_test.reshape(-1,1)), 1), batch_size=64, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.line1 = nn.Linear(784, 100)
        self.line2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = F.relu(self.line1(x))
        # x = F.dropout(x, 0.2, training=self.training)
        x = self.line2(x)
        return x

net = Net()
optim = torch.optim.Adam(params=net.parameters())
loss_function = nn.CrossEntropyLoss()

def train_model(num, save_model=True):
    tmp_loss = []
    tmp_correct = []
    for epoch in range(num):
        net.train()
        running_loss = 0
        for data in train_loader:
            input, label = data[:,:-1].float(), data[:, -1].long()
            output = net(input)
            loss = loss_function(output, label)

            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()
        print("loss:{}".format(running_loss))
        tmp_loss.append(running_loss)

        net.eval()
        test_correct = 0
        test_1 = 0
        for data in test_loader:
            input, label = data[:, :-1].float(), data[:, -1].long()
            output = net(input)
            _, output = output.max(dim=1)
            test_correct += (label == output).sum().item()
            test_1 += label.shape[0]
        right_rate = round(test_correct/len(y_test) * 100.0, 2)
        tmp_correct.append(right_rate)
        print("正确率:{}%".format(right_rate))

    torch.save(net, 'net_model.pkl')
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(tmp_loss, label='loss')
    plt.plot(tmp_correct, label='acc(%)')
    plt.legend()

    # 必须保存在当前目录下，可以保存多张图片
    # 推荐使用 svg 格式，目前支持 svg、png、jpg jpeg 格式
    plt.savefig('report.svg')


if __name__ == '__main__':
    train_model(10)
