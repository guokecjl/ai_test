# coding: utf-8
"""https://github.com/apachecn/AiLearning/blob/master/docs/ml/8.md"""

import os
import sys
sys.path.append(os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".."))

import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(200, 1)
y = np.random.rand(200, 1) * np.random.randint(8, 10, (200, 1))/100.0 + x


def stand_regress(x, y):
    """
    线性回归
    :param x:　自变量x
    :param y: 　对于输入数据的输出y
    :return: ws系数矩阵
    """
    x = np.mat(x)
    y = np.mat(y)
    x_t_x = np.dot(x.T, x)
    if not np.linalg.det(x_t_x):
        print('x.T * x的逆矩阵不存在')
        return
    ws = np.dot(x_t_x.I, np.dot(x.T, y))
    return ws

if __name__ == '__main__':
    ws = stand_regress(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x[:,0].flatten(), y[:, 0].flatten())
    x_copy = x.copy()
    x.sort(axis=0)
    y_hat = x_copy * ws
    ax.plot(x_copy[:, 0], y_hat)
    plt.show()