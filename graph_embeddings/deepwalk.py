# coding: utf-8

"""
图嵌入算法的一种实现
"""

import dgl
import pandas as pd
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

import sys
import os
base_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "..")
sys.path.append(base_dir)

edges = pd.read_table('{}/data/Wiki_edgelist.txt'.format(base_dir), sep=' ')
nodes = pd.read_table('{}/data/wiki_labels.txt'.format(base_dir), sep=' ')

u = edges['src'].to_numpy()
v = edges['dst'].to_numpy()
labels = nodes['label'].to_numpy()

g = dgl.graph((u, v))
g.ndata['label'] = th.tensor(labels)


class DeepWalk(object):
    def __init__(self, graph, walk_length, num_walks):

        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}
        self.sentences = self.get_sentences(walk_length, num_walks)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            word = str(word.item())
            if word in self.w2v_model.wv:
                self._embeddings[word] = self.w2v_model.wv[word]
        return self._embeddings

    def get_sentences(self, walk_length, num_walks):
        walks = dgl.sampling.random_walk(
            self.graph, th.randint(0, self.graph.num_nodes(), (num_walks,)), length=walk_length)
        return list(map(lambda item: list(map(str, item)), walks[0].tolist()))


def plot_embeddings(embeddings):
    emb_list = []
    for i in range(len(labels)):
        if str(i) in embeddings:
            emb_list.append(embeddings[str(i)])
        else:
            emb_list.append(np.ones(128))

    emb_list = np.array(emb_list)
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(labels)):
        if str(i) not in embeddings:
            continue
        color_idx.setdefault(labels[i].item(), [])
        color_idx[labels[i].item()].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


model = DeepWalk(g, 10, 3644)
model.train(window_size=5, iter=5)
embeddings = model.get_embeddings()
plot_embeddings(embeddings)