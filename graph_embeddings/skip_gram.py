# coding: utf-8

"""
数据集包含 2,405 个网页和17,981条网页之间的链接关系，以及每个网页的所属类别
通过dgl对数据进行采样，然后使用自己实现的word2vec算法实现对节点的聚类
"""

import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random

# 增加项目路径
import sys
import os
base_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "..")
sys.path.append(base_dir)

from graph_embeddings.plot_embedding import plot_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.set_num_threads(2)

edges = pd.read_table('{}/data/Wiki_edgelist.txt'.format(base_dir), sep=' ')
nodes = pd.read_table('{}/data/wiki_labels.txt'.format(base_dir), sep=' ')

u = edges['src'].to_numpy()
v = edges['dst'].to_numpy()
labels = nodes['label'].to_numpy()

g = dgl.graph((u,v))
g.ndata['label'] = torch.tensor(labels)

num_node = g.num_nodes()
C = 5  # context window
simple_num = 10000
walks = dgl.sampling.random_walk(g, torch.randint(0, 2405, (simple_num, )), length=C * 2)
# 过滤掉-1的节点,-1表示找不到下一条边
walks = list(filter(lambda item: (item < 0).sum().item() == 0, walks[0]))
walks = np.array(list(map(lambda item: item.tolist(), walks)))
walks_train = np.delete(walks, C, axis=1).reshape(-1, C * 2)
walks_label = walks[:, C:C+1].reshape(-1, 1)

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

K = 1  # number of negative samples
epochs = 2
MAX_VOCAB_SIZE = 2405
EMBEDDING_SIZE = 128
batch_size = 32
lr = 0.2


class WordEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(WordEmbeddingDataset, self).__init__()  # #通过父类初始化模型，然后重写两个方法
        self.word_freqs = torch.ones(2405)/2405.0

    def __len__(self):
        return len(walks)  # 返回所有单词的总数，即item的总数

    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的positive word
            - 随机采样的K个单词作为negative word
        '''
        center_words = walks_label[idx].item()  # 取得中心词
        pos_words = torch.tensor(walks_train[idx])  # 先取得中心左右各C个词的索引
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0],
                                   True)

        # while 循环是为了保证 neg_words中不能包含背景词
        while len(set(pos_words) & set(neg_words)) > 0:
            neg_words = torch.multinomial(self.word_freqs,
                                          K * pos_words.shape[0], True)

        return center_words, pos_words, neg_words


class EmbeddingModel(nn.Module):
    """
    原生的word2vec算法实现
    """
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size # 单词数量
        self.embed_size = embed_size # 编码后的维度

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        input_embedding = self.in_embed(
            input_labels)  # [batch_size, embed_size]
        pos_embedding = self.out_embed(
            pos_labels)  # [batch_size, 2 * C, embed_size]
        neg_embedding = self.out_embed(
            neg_labels)  # [batch_size, C * 2 * K, embed_size]

        input_embedding = input_embedding.unsqueeze(
            2)  # [batch_size, embed_size, 1]

        pos_dot = torch.bmm(pos_embedding,
                            input_embedding)  # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2)  # [batch_size, (window * 2)]

        neg_dot = torch.bmm(neg_embedding,
                            -input_embedding)  # [batch_size, C * 2 * K, 1]
        neg_dot = neg_dot.squeeze(2)  # batch_size, C * 2 * K]

        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + log_neg

        return -loss

    def input_embedding(self):
        return self.in_embed.weight.cpu().detach().numpy()


class EmbeddingModelLiner(nn.Module):
    """
    修改了原生的方法，使用全连接代替原始mean操作，使用cos_loss代替log_sigmoid
    """
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModelLiner, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.loss_func = nn.CosineEmbeddingLoss() # cos_loss用来衡量相似度

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.line1 = nn.Linear(2 * C * EMBEDDING_SIZE, self.embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        input_embedding = self.in_embed(
            input_labels)  # [batch_size, embed_size]
        pos_embedding = self.out_embed(
            pos_labels)  # [batch_size, C * 2, embed_size]
        neg_embedding = self.out_embed(
            neg_labels)  # [batch_size, C * 2, embed_size]

        pos_embedding = pos_embedding.reshape(
            -1, 2 * C * EMBEDDING_SIZE) # [batch_size, C * 2 * embed_size]
        neg_embedding = neg_embedding.reshape(
            -1, 2 * C * EMBEDDING_SIZE) # [batch_size, C * 2 * embed_size]

        pos_embedding = self.line1(pos_embedding) # [b, embed_size]
        neg_embedding = self.line1(neg_embedding) # [b, embed_size]

        log_pos = self.loss_func(input_embedding, pos_embedding,
                                 target=torch.ones(input_embedding.shape[0]))
        log_neg = self.loss_func(input_embedding, neg_embedding,
                                 target=-torch.ones(input_embedding.shape[0]))

        loss = log_pos + log_neg
        return loss

    def input_embedding(self):
        return self.in_embed.weight.cpu().detach().numpy()

data_set = WordEmbeddingDataset()
data_loader = torch.utils.data.DataLoader(data_set, batch_size, shuffle=True)

model = EmbeddingModelLiner(MAX_VOCAB_SIZE, EMBEDDING_SIZE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(500):
    model.train()
    total_loss = 0
    for i, (input_labels, pos_labels, neg_labels) in enumerate(data_loader):
        input_labels = input_labels.long().to(device)
        pos_labels = pos_labels.long().to(device)
        neg_labels = neg_labels.long().to(device)

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        total_loss += loss.item()
        loss.backward()

        optimizer.step()

    print('epoch:{}, loss: {}'.format(epoch, total_loss))

    model.eval()
    if epoch % 20 == 19:
        embed = model.input_embedding()
        plot_embeddings(embed, labels, path='./tmp/epoch_{}.png'.format(epoch))
