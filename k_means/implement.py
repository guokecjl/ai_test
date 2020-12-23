# coding: utf-8
"""二分k-means算法"""

import math
import os
import sys
sys.path.append(os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".."))

import numpy as np
import pandas as pd
data_set = pd.read_csv('{}/tan_day_std.csv'.format(os.path.dirname(
    os.path.abspath(__file__))))
mid_month = data_set.pop('mid_month')
data_set = data_set.to_numpy()


def cal_distance(v_a, v_b):
    """
    计算距离
    """
    return math.sqrt(sum(pow(v_a -v_b, 2)))


def rand_cent(data_set, k):
    """
    随机生成k个质心
    """
    feat = data_set.shape[1]
    cent_roids = np.zeros((k, feat))
    for j in range(feat):
        min_j = min(data_set[:, j])
        range_j = float(max(data_set[:, j]) - min_j)
        cent_roids[:, j] = min_j + range_j * np.random.rand(k)
    return cent_roids


def k_means(data_set, k):
    """
    该算法会随机创建k个质心，然后将每个点分配到距离最近的质心，再重新计算质心
    """
    num = data_set.shape[0]
    rand_roids = rand_cent(data_set, k)
    node_dist = np.zeros((num, 2))
    symbol = True
    while symbol:
        symbol = False
        for i in range(num):
            dist = -1
            min_index = -1
            for j, center in enumerate(rand_roids):
                new_dist = cal_distance(center, data_set[i])
                if dist == -1 or new_dist < dist:
                    dist = new_dist
                    min_index = j
            if node_dist[i, 0] != min_index:
                node_dist[i, :] = min_index, dist ** 2
                symbol = True
            if min_index == 0:
                node_dist[i, 1] = dist ** 2
        for cent in range(k):
            node_in_roid = data_set[node_dist[:, 0] == cent]
            if node_in_roid.size == 0:
                continue
            rand_roids[cent, :] = np.mean(node_in_roid, axis=0)
    return rand_roids, node_dist


tmp = 0
for i in range(50):
    rand_roids, node_dist = k_means(data_set[:,:-1], 2)
    sse = node_dist[:, 1].sum()
    if tmp == 0 or sse < tmp:
        tmp = sse
        # tmp_data = pd.DataFrame(node_dist)
        # tmp_data[2] = mid_month
        # tmp_data.to_csv('tmp.csv', index=False)
        print(tmp)
