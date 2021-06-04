# coding: utf-8

'''
通过dnn识别手写数字集
'''
import os
base_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

data_train = torchvision.datasets.MNIST(root = "/home/zhouxi/ai",
                            transform=torchvision.transforms.ToTensor(),
                            train=True,
                            download=False)
loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True)

data_test = torchvision.datasets.MNIST(root = "/home/zhouxi/ai",
                            transform=torchvision.transforms.ToTensor(),
                            train=False,
                            download=False)
loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                batch_size=64,
                                                shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.line1 = nn.Linear(784, 100)
        self.line2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = F.relu(self.line1(x))
        x = F.dropout(x, 0.2)
        x = self.line2(x)
        return x


def train_model(num, save_model=True):
    net = Net()
    optim = torch.optim.Adam(params=net.parameters())
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(num):
        net.train()
        running_loss = 0
        for data, label in loader_train:
            output = net(data)
            loss = loss_function(output, label)

            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()
        print("loss:{}".format(running_loss))

        net.eval()
        test_correct = 0
        for data, label in loader_test:
            output = net(data)
            _, output = output.max(dim=1)
            test_correct += (label == output).sum().item()
        print("正确率:{}%".format(round(test_correct/100.0, 2)))

    if save_model:
        # 保存模型
        torch.save(net, os.path.join(base_dir, 'test.pth.tar'))


def use_save_model():
    # 加载模型
    net = torch.load(os.path.join(base_dir, 'test.pth.tar'))
    test_correct = 0
    for data, label in loader_test:
        output = net(data)
        _, output = output.max(dim=1)
        test_correct += (label == output).sum().item()
    print("存储的模型分类正确率:{}%".format(round(test_correct / 100.0, 2)))


if __name__ == '__main__':
    train_model(20, False)
    # use_save_model()
