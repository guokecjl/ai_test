# coding: utf-8

'''
手动实现word2vec逻辑
'''

import dgl
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

edges = pd.read_table('Wiki_edgelist.txt', sep=' ')
nodes = pd.read_table('wiki_labels.txt', sep=' ')

u = edges['src'].to_numpy()
v = edges['dst'].to_numpy()
labels = nodes['label'].to_numpy()

g = dgl.graph((u,v))
g.ndata['label'] = th.tensor(labels)

num_node = g.num_nodes()
walk_len = 10
walks = dgl.sampling.random_walk(g, th.arange(0, num_node).repeat(2), length=walk_len)
walks = list(filter(lambda item: (item < 0).sum().item() == 0, walks[0]))
walks = np.array(list(map(lambda item: item.tolist(), walks)))
walks_train = np.delete(walks, int(walk_len/2), axis=1).reshape(-1, walk_len)
walks_label = walks[:, int(walk_len/2):int(walk_len/2)+1].reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(walks_train, walks_label, test_size=0.2, random_state=42)
x_test = th.tensor(x_test)
y_test = th.tensor(y_test)
train_loader = th.utils.data.DataLoader(np.concatenate((x_train, y_train), 1), batch_size=64, drop_last=True, shuffle=True)


class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.word_emb = nn.Embedding(2405, 128)
        self.line1 = nn.Linear(1280, 128)
        self.line2 = nn.Linear(128, 2405)

    def forward(self, x):
        x = self.word_emb(x)
        x = x.reshape(-1, 1280)
        embed = F.relu(self.line1(x))
        x = self.line2(embed)
        return x, embed


class Word2Vec1(nn.Module):
    def __init__(self):
        super(Word2Vec1, self).__init__()
        self.line1 = nn.Linear(2405 * 10, 2405)
        self.line2 = nn.Linear(2405, 128)

    def forward(self, x):
        x = F.relu(self.line1(x))
        x = F.dropout(x, 0.1, training=self.training)
        x = F.logsigmoid(self.line2(x))
        return x


class Word2Vec2(nn.Module):
    def __init__(self):
        super(Word2Vec2, self).__init__()
        self.line1 = nn.Linear(2405, 128)

    def forward(self, x):
        x = F.logsigmoid(self.line1(x))
        return x


def plot_embeddings(embeddings, emb_list, acc):
    """
    将转换后的向量降到二位空间，并绘制成图
    """
    print('plotting')

    print(emb_list.shape)
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
    plt.xlim(xmin=-15)
    plt.xlim(xmax=15)
    plt.ylim(ymin=-15)
    plt.ylim(ymax=15)
    plt.legend()
    plt.savefig('./tmp/{}_acc{}.png'.format(epoch, acc))
    plt.close()



# net = Word2Vec()
# loss_func = nn.CrossEntropyLoss()
# optim = th.optim.Adam(params=net.parameters(), lr=0.0005)
#
# for epoch in range(50):
#     net.train()
#     total_loss = 0
#     for data in train_loader:
#         input, label = data[:, :-1].long(), data[:, -1].long()
#         output, _ = net(input)
#         loss = loss_func(output, label)
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
#         total_loss += loss.item()
#
#     net.eval()
#     input = th.tensor(x_train)
#     label = y_train
#     eval_label, embed = net(input.long())
#     _, eval_label = eval_label.max(axis=1)
#     label = th.tensor(label.flatten())
#     acc_rate = (eval_label == label).sum().item()/len(label)
#     print("epoch: {}, loss:{}, 正确率:{}".format(epoch, total_loss, acc_rate))
#
#     embeddings = {}
#     for i, item in enumerate(label):
#         embeddings[str(item.item())] = embed[i].detach().numpy()
#     if epoch % 10 == 9:
#         plot_embeddings(embeddings=embeddings, acc=acc_rate)


# class SkipGram(nn.Module):
#     def __init__(self, vocab_size, embed_size):
#         super(SkipGram, self).__init__()
#         initrange = 0.5 / embed_size
#         self.u_embedding_matrix = nn.Embedding(vocab_size, embed_size)
#         self.u_embedding_matrix.weight.data.uniform_(-initrange, initrange)
#         self.v_embedding_matrix = nn.Embedding(vocab_size, embed_size)
#         self.v_embedding_matrix.weight.data.uniform_(-0, 0)
#
#     def forward(self, pos_u, pos_v, neg_u, neg_v):
#         embed_pos_u = self.v_embedding_matrix(pos_u)
#         embed_pos_v = self.u_embedding_matrix(pos_v)
#         score = th.mul(embed_pos_u, embed_pos_v)
#         score = th.sum(score, dim=1)
#         log_target = F.logsigmoid(score).squeeze()
#
#         embed_neg_u = self.u_embedding_matrix(neg_u)
#         embed_neg_v = self.v_embedding_matrix(neg_v)
#
#         neg_score = th.mul(embed_neg_u, embed_neg_v)
#         neg_score = th.sum(neg_score, dim=1)
#         sum_log_sampled = F.logsigmoid(-1 * neg_score).squeeze()
#
#         loss = log_target.sum() + sum_log_sampled.sum()
#         loss = -1 * loss
#         return loss
# net1 = Word2Vec1()
# net2 = Word2Vec2()
# optim1 = th.optim.Adam(net1.parameters())
# optim2 = th.optim.Adam(net2.parameters())
# loss_func = nn.MSELoss()
#
# net1.train()
# net2.train()
#
# for epoch in range(2):
#     total_loss = 0
#     for data in train_loader:
#         input = nn.functional.one_hot(data[:, :-1], 2405).float().reshape(-1, 24050)
#         t_label = nn.functional.one_hot(data[:, -1], 2405).float().reshape(-1, 2405)
#         f_label = nn.functional.one_hot(th.randint(0, 2405, (t_label.shape[0], )), 2405).float().reshape(-1, 2405)
#         output1 = net1(input)
#         t_output2 = net2(t_label)
#         f_output2 = net2(f_label)
#
#         optim1.zero_grad()
#         optim2.zero_grad()
#         loss1 = loss_func(output1, t_output2)
#         loss2 = loss_func(output1, f_output2)
#         loss = - (loss1 + loss2)
#         loss.backward()
#         optim1.step()
#         optim2.step()
#         total_loss += loss
#     print(total_loss)
#
# net1.eval()
# eval_y = net1(nn.functional.one_hot(x_test, 2405).float().reshape(-1, 24050))
# y_test = y_test.flatten()
# embeddings = {}
# for i, item in enumerate(y_test):
#     embeddings[str(item.item())] = eval_y[i].detach().numpy()
#
# plot_embeddings(embeddings=embeddings)


class EmbeddingVec(nn.Module):
    def __init__(self):
        super(EmbeddingVec, self).__init__()
        self.in_size = 2405
        self.emb_size = 128

        self.in_emb = nn.Embedding(self.in_size, self.emb_size) # [b]
        self.out_emb = nn.Embedding(self.in_size, self.emb_size) # [b, 4]

    def forward(self, input_labels, pos_labels, neg_lables):
        input_emb = self.in_emb(input_labels) # [b, emb_size]
        pos_emb = self.out_emb(pos_labels) # [b, 4, emb_size]
        nag_emb = self.out_emb(neg_lables) # [b, 4 * k, emb_size]
        input_emb = input_emb.repeat(walk_len, 1)
        pos_emb = pos_emb.reshape(-1, self.emb_size)
        nag_emb = nag_emb.reshape(-1, self.emb_size)

        return input_emb, pos_emb, nag_emb

    def get_emb(self):
        return self.in_emb.weight.detach().numpy()

K = 1
class WordEmbeddingDataset(th.utils.data.Dataset):
    def __init__(self):
        super(WordEmbeddingDataset, self).__init__()
        self.word_freqs = th.ones(2405)/2405.0

    def __len__(self):
        return len(walks)  # 返回所有单词的总数，即item的总数

    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的positive word
            - 随机采样的K个单词作为negative word
        '''

        center_words = walks_label[idx]  # 取得中心词
        pos_words = th.tensor(walks_train[idx])  # 先取得中心左右各C个词的索引

        neg_words = th.multinomial(self.word_freqs, K * pos_words.shape[0], True)
        # while 循环是为了保证 neg_words中不能包含背景词
        while len(set(pos_words.numpy().tolist()) & set(
                neg_words.numpy().tolist())) > 0:
            neg_words = th.multinomial(self.word_freqs,
                                          K * pos_words.shape[0], True)

        return center_words, pos_words, neg_words

dataset = WordEmbeddingDataset()
dataloader = th.utils.data.DataLoader(dataset, 64, shuffle=True)

net = EmbeddingVec()
loss_func = nn.CosineEmbeddingLoss()
optim = th.optim.Adam(params=net.parameters(), lr=0.01)


for epoch in range(500):
    net.train()
    total_loss = 0
    total_loss_pos = 0
    total_loss_neg = 0
    x = 0
    for data in dataloader:
        x += 64
        input, pos, nag = data
        input_emb, pos_emb, nag_emb = net(input.reshape(-1), pos, nag)
        loss_pos = loss_func(input_emb, pos_emb, th.ones(64 * walk_len, 1))
        loss_neg = loss_func(input_emb, nag_emb, -th.ones(64 * walk_len, 1))

        total_loss_pos += loss_pos
        total_loss_neg += loss_neg

        loss = loss_pos + loss_neg
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item()
    print(x, len(walks))

    # net.eval()
    # eval_y, embed = net(x_test.long())
    # _, eval_y = eval_y.max(axis=1)
    # y_test = y_test.flatten()
    # acc_rate = (eval_y == y_test).sum().item()/len(y_test)
    print("epoch: {}, loss:{}, loss_pos:{}, loss_neg:{}".format(epoch, total_loss, total_loss_pos, total_loss_neg))

    if epoch % 20 == 19:
        embed = net.get_emb()
        embeddings = {}
        for i in range(embed.shape[0]):
            embeddings[str(i)] = embed[i]
        plot_embeddings(embeddings, embed, acc=1)
