# coding: utf-8

import dgl
import torch

import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F


g = dgl.data.CoraGraphDataset()[0]

sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
dataloader = dgl.dataloading.NodeDataLoader(
    g, torch.arange(g.num_nodes()), sampler,
    batch_size=512,
    shuffle=True,
    drop_last=False,
    num_workers=4)

class TwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_features, hidden_features)
        self.conv2 = dglnn.GraphConv(hidden_features, out_features)

    def forward(self, g, x):
        x = F.relu(self.conv1(g, x))
        x = F.relu(self.conv2(g, x))
        return x

class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
        self.conv2 = dgl.nn.GraphConv(hidden_features, out_features)

    def forward(self, blocks, x):
        x = F.relu(self.conv1(blocks[0], x))
        x = F.relu(self.conv2(blocks[1], x))
        return x

in_features, hidden_features, out_features = 300, 50, 23
model = StochasticTwoLayerGCN(in_features, hidden_features, out_features)
opt = torch.optim.Adam(model.parameters())

compute_loss = F.cross_entropy

for epoch in range(1):
    loss_sum = 0
    for input_nodes, output_nodes, blocks in dataloader:
        blocks = [b for b in blocks]
        input_features = blocks[0].srcdata['feat']
        output_labels = blocks[-1].dstdata['label']
        output_predictions = model(blocks, input_features)
        print(output_predictions.shape)
        loss = compute_loss(output_predictions, output_labels)
        opt.zero_grad()
        loss.backward()
        loss_sum += loss.item()
        opt.step()
    print(epoch, loss_sum)
