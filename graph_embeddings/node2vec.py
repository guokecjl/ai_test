# coding: utf-8

import random
import itertools
import dgl
import pandas as pd
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from joblib import Parallel, delayed

edges = pd.read_table('data/Wiki_edgelist.txt', sep=' ')
nodes = pd.read_table('data/wiki_labels.txt', sep=' ')

u = edges['src'].to_numpy()
v = edges['dst'].to_numpy()
labels = nodes['label'].to_numpy()
# u = th.randint(0, 10, (100, ))
# v = th.randint(0, 10, (100, ))
# labels = th.randint(0, 2, (10, ))

g = dgl.graph((u,v))
g.ndata['label'] = th.tensor(labels)


class RandomWalker:
    def __init__(self, g, p=1, q=1, workers=1, use_rejection_sampling=0):
        """
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        :param use_rejection_sampling: Whether to use the rejection sampling strategy in node2vec.
        """
        self.g = g
        self.p = p
        self.q = q
        self.use_rejection_sampling = use_rejection_sampling

    def node2vec_walk(self, walk_length, start_node):

        g = self.g
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = g.successors(cur).tolist()
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[self.alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[self.alias_sample(alias_edges[edge][0],
                                                      alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break

        return walk

    def node2vec_walk2(self, walk_length, start_node):
        """
        Reference:
        KnightKing: A Fast Distributed Graph Random Walk Engine
        http://madsys.cs.tsinghua.edu.cn/publications/SOSP19-yang.pdf
        """

        def rejection_sample(inv_p, inv_q, nbrs_num):
            upper_bound = max(1.0, max(inv_p, inv_q))
            lower_bound = min(1.0, min(inv_p, inv_q))
            shatter = 0
            second_upper_bound = max(1.0, inv_q)
            if (inv_p > second_upper_bound):
                shatter = second_upper_bound / nbrs_num
                upper_bound = second_upper_bound + shatter
            return upper_bound, lower_bound, shatter

        G = self.G
        alias_nodes = self.alias_nodes
        inv_p = 1.0 / self.p
        inv_q = 1.0 / self.q
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[self.alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    upper_bound, lower_bound, shatter = rejection_sample(
                        inv_p, inv_q, len(cur_nbrs))
                    prev = walk[-2]
                    prev_nbrs = set(G.neighbors(prev))
                    while True:
                        prob = random.random() * upper_bound
                        if (prob + shatter >= upper_bound):
                            next_node = prev
                            break
                        next_node = cur_nbrs[self.alias_sample(
                            alias_nodes[cur][0], alias_nodes[cur][1])]
                        if (prob < lower_bound):
                            break
                        if (prob < inv_p and next_node == prev):
                            break
                        _prob = 1.0 if next_node in prev_nbrs else inv_q
                        if (prob < _prob):
                            break
                    walk.append(next_node)
            else:
                break
        return walk

    def get_alias_edge(self, t, v):
        """
        compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
        :param t:
        :param v:
        :return:
        """
        g = self.g
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in g.successors(v):
            # weight = G[v][x].get('weight', 1.0)  # w_vx
            weight = 1
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight/p)
            elif g.has_edges_between(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return self.create_alias_table(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        g = self.g
        alias_nodes = {}
        for node_id in g.nodes():
            node_id = node_id.item()
            # unnormalized_probs = [G[node][nbr].get('weight', 1.0)
            #                       for nbr in G.neighbors(node)]
            unnormalized_probs = [1.0 for nbr in g.successors(node_id)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node_id] = self.create_alias_table(normalized_probs)

        if not self.use_rejection_sampling:
            alias_edges = {}

            src_node, dst_node = g.edges()
            all_edges = zip(src_node.tolist(), dst_node.tolist())
            for edge in all_edges:
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            self.alias_edges = alias_edges

        self.alias_nodes = alias_nodes
        return

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):

        g = self.g

        nodes = g.nodes().tolist()

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            self.partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length,):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                if self.p == 1 and self.q == 1:
                    # walks.append(self.deepwalk_walk(
                    #     walk_length=walk_length, start_node=v))
                    pass
                elif self.use_rejection_sampling:
                    walks.append(self.node2vec_walk2(
                        walk_length=walk_length, start_node=v))
                else:
                    walks.append(self.node2vec_walk(
                        walk_length=walk_length, start_node=v))
        return walks

    @staticmethod
    def create_alias_table(area_ratio):
        """
        :param area_ratio: sum(area_ratio)=1
        :return: accept,alias
        """
        l = len(area_ratio)
        accept, alias = [0] * l, [0] * l
        small, large = [], []
        area_ratio_ = np.array(area_ratio) * l
        for i, prob in enumerate(area_ratio_):
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            small_idx, large_idx = small.pop(), large.pop()
            accept[small_idx] = area_ratio_[small_idx]
            alias[small_idx] = large_idx
            area_ratio_[large_idx] = area_ratio_[large_idx] - \
                                     (1 - area_ratio_[small_idx])
            if area_ratio_[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)

        while large:
            large_idx = large.pop()
            accept[large_idx] = 1
        while small:
            small_idx = small.pop()
            accept[small_idx] = 1

        return accept, alias

    @staticmethod
    def alias_sample(accept, alias):
        """
        :param accept:
        :param alias:
        :return: sample index
        """
        N = len(accept)
        i = int(np.random.random() * N)
        r = np.random.random()
        if r < accept[i]:
            return i
        else:
            return alias[i]

    @staticmethod
    def partition_num(num, workers):
        if num % workers == 0:
            return [num // workers] * workers
        else:
            return [num // workers] * workers + [num % workers]


class Node2Vec(object):
    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1, use_rejection_sampling=0):
        self.graph = graph
        self._embeddings = {}
        self.w2v_model = None
        self.sentences = self.get_sentences(walk_length, num_walks, p, q, workers, use_rejection_sampling)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model

    def get_embeddings(self):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            word = str(word.item())
            if word in self.w2v_model.wv:
                self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings

    def get_sentences(self, walk_length, num_walks, p, q, workers, use_rejection_sampling):
        random_work = RandomWalker(
            self.graph, p=p, q=q, workers=workers, use_rejection_sampling=use_rejection_sampling)
        random_work.preprocess_transition_probs()
        walks = random_work.simulate_walks(num_walks=num_walks,
                                           walk_length=walk_length, workers=workers, verbose=1)
        walks = np.array(walks)
        # np.save('walks.npy', walks)
        # walks = np.load('walks.npy', allow_pickle=True)
        return list(map(lambda item: list(map(str, item)), walks))


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

model = Node2Vec(g, walk_length=10, num_walks=80, p=0.25, q=4, workers=2, use_rejection_sampling=0)
model.train(window_size=5, iter=3)
embeddings = model.get_embeddings()
plot_embeddings(embeddings)
