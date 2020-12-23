# coding: utf-8
"""
https://zhuanlan.zhihu.com/p/139617364
https://zhuanlan.zhihu.com/p/79064602
"""

import torch

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


def load_data():
    seq_number = np.array(
        [112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104.,
         118., 115., 126., 141., 135., 125., 149., 170., 170., 158., 133.,
         114., 140., 145., 150., 178., 163., 172., 178., 199., 199., 184.,
         162., 146., 166., 171., 180., 193., 181., 183., 218., 230., 242.,
         209., 191., 172., 194., 196., 196., 236., 235., 229., 243., 264.,
         272., 237., 211., 180., 201., 204., 188., 235., 227., 234., 264.,
         302., 293., 259., 229., 203., 229., 242., 233., 267., 269., 270.,
         315., 364., 347., 312., 274., 237., 278., 284., 277., 317., 313.,
         318., 374., 413., 405., 355., 306., 271., 306., 315., 301., 356.,
         348., 355., 422., 465., 467., 404., 347., 305., 336., 340., 318.,
         362., 348., 363., 435., 491., 505., 404., 359., 310., 337., 360.,
         342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405.,
         417., 391., 419., 461., 472., 535., 622., 606., 508., 461., 390.,
         432.], dtype=np.float32)
    seq_year = np.repeat(np.arange(12), 12).reshape(144, 1)
    seq_month = np.tile(np.arange(12), 12).reshape(144, 1)
    seq_number = seq_month.reshape(seq_number.shape[0], 1)
    seq = np.concatenate((seq_year, seq_month, seq_number), axis=1)
    # normalization
    seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)
    return seq


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * 3, output_size)

    def forward(self, x):
        x, _ = self.lstm(x) # x is input, size (seq_len, batch, input_size)
        b, s, h = x.shape # x is output, size (seq_len, batch, hidden_size)
        x = x.reshape(b, s * h)
        x = self.fc(x)
        return x


def create_dataset(seq, step):
    x, y = [], []
    for i in range(len(seq) - step):
        x.append(seq[i:i+step, 1:])
        y.append(seq[i+step, -1])
    return torch.Tensor(x), torch.Tensor(y)


if __name__ == '__main__':
    data_set = load_data()
    train_size  = int(len(data_set) * 0.7)
    x, y = create_dataset(data_set, 3)
    train_x, test_x = x[:train_size], x[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    train_x = train_x.reshape(train_x.shape[0], -1, 2)

    train_y = train_y.reshape(train_y.shape[0], 1)
    net = Net(input_size=2, hidden_size=8)
    optim = torch.optim.Adam(net.parameters(), lr=0.02)
    loss_func = nn.MSELoss()
    net.train()
    for epoch in range(2):
        out = net(train_x)
        optim.zero_grad()
        loss = loss_func(out, train_y)
        loss.backward()
        optim.step()
        if epoch % 10 == 9:
            print(loss.item())

    net.eval()
    test_x = test_x.reshape(test_x.shape[0], -1 , 2)
    eval_y = net(test_x)
    eval_y = eval_y.flatten()

    plt.figure()
    plt.plot(eval_y.flatten().tolist(), 'r')
    plt.plot(test_y.flatten().tolist(), 'b')
    plt.show()