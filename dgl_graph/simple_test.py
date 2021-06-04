# coding: utf-8

import dgl
import torch
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

dataset = dgl.data.CiteseerGraphDataset()
g = dataset[0]
node_features = g.ndata['feat']
node_labels = g.ndata['label']
train_mask = g.ndata['train_mask']
valid_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']
n_feat = node_features.shape[1]
n_labels = len(node_labels.unique())


class SAGE(nn.Module):

    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h


def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        output = model(graph, features)
        r_label = labels[mask]
        output = output[mask]
        _, e_label = torch.max(output, dim=1)
        correct_num = (r_label == e_label).sum().item()
        print("正确率:{}%".format(correct_num * 100.0/len(r_label)))


model = SAGE(in_feats=n_feat, hid_feats=100, out_feats=n_labels)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    output = model(g, node_features)
    r_lable = node_labels[train_mask]
    output = output[train_mask]

    optim.zero_grad()
    loss = loss_func(output, r_lable)
    loss.backward()
    optim.step()
    evaluate(model, g, node_features, node_labels, test_mask)
