# coding: utf-8

import torch as th
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_embeddings(emb_list, labels, path):
    """
    将转换后的向量降到二位空间，并绘制成图
    """
    print('plotting')

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)
    # 过滤掉一些偏离中心很远的节点
    for xx in range(20):
        node_pos[node_pos.argmax(0)] = th.zeros(2, 2)
    for xx in range(20):
        node_pos[node_pos.argmin(0)] = th.zeros(2, 2)

    color_idx = {}
    for i in range(emb_list.shape[0]):
        color_idx.setdefault(labels[i].item(), [])
        color_idx[labels[i].item()].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c,s=10)

    plt.legend()
    plt.savefig(path, dpi=350)
    plt.close()
