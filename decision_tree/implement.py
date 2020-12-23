# coding: utf-8

import math
import os
import sys
sys.path.append(os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".."))

import numpy as np
from decision_tree import tree_plot

data_set = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
data_set = np.array(data_set)
labels = ['no surfacing', 'flippers']


def cal_shannon_ent(data_set):
    """
    h =+ -p(i)log(p(i), 2)
    """
    label = data_set[:, -1]
    shannon_ent = 0
    for item in set(label):
        label_count = (label == item).sum().item()
        prob = label_count / float(len(label))
        shannon_ent -= prob * math.log(prob, 2)
    return shannon_ent


def split_data_set(data_set, index, value):
    """
    返回data_set中index列=value的所有行,该行已经去掉了index列
    """
    index_list = data_set[data_set[:, index] == value]
    tmp_data = np.delete(index_list, index, axis=1)
    return tmp_data


def choose_best_feature(data_set):
    """
    选择最好的特征
    """
    len_feature = data_set.shape[1] - 1
    base_entropy = cal_shannon_ent(data_set)
    best_gain, best_feature = 0.0, -1
    for i in range(len_feature):
        feat_list = data_set[:,i]
        unique_val = set(feat_list)
        new_entropy = 0.0
        for val in unique_val:
            sub_data_set = split_data_set(data_set, i, val)
            prob = len(sub_data_set)/float(len(data_set))
            new_entropy += prob * cal_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if best_gain < info_gain:
            best_gain = info_gain
            best_feature = i
    return best_feature


def create_tree(data_set, labels):
    """
    训练树
    :return:
    """
    sub_labels = labels[:]
    if len(set(data_set[:, -1])) == 1:
        return data_set[0][-1]
    if len(data_set[0]) == 1:
        raise Exception

    best_feat = choose_best_feature(data_set)
    best_feat_label = sub_labels[best_feat]
    my_tree = {best_feat_label: {}}
    sub_labels.pop(best_feat)
    feat_value = data_set[:, best_feat]
    for value in set(feat_value):
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


def clasify(data, labels, my_tree):
    """
    :param data: 被测试的数据
    :param labels: 标签
    :param my_tree: 树
    :return: 标签
    """

    first_key = list(my_tree.keys())[0]
    label_index = labels.index(first_key)
    second_key = data[label_index]
    new_tree = my_tree[first_key][second_key]
    if isinstance(new_tree, dict):
        return clasify(data, labels, new_tree)
    else:
        return new_tree

my_tree = create_tree(data_set, labels)
tree_plot.createPlot(my_tree)
print(clasify(data_set[3], labels, my_tree) == data_set[3][-1])