# coding: utf-8

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


class ConvNet(nn.Module):
    """

    """
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.line1 = nn.Linear(64 * 5 * 5, 100)
        self.line2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape(-1, 64 * 5 * 5)
        x = F.relu(self.line1(x))
        x = F.dropout(x, 0.2)
        x = F.relu(self.line2(x))
        return x

net = ConvNet()
optim = torch.optim.Adam(params=net.parameters(), lr=0.005)
loss_func = nn.CrossEntropyLoss()

for epoch in range(10):
    net.train()
    running_loss = 0
    for data, label in loader_train:
        output = net(data)
        loss = loss_func(output, label)

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
