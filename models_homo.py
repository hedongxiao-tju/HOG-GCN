import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphConvolution_homo
from torch.nn.parameter import Parameter
import torch
import math


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x)

    def get_emb(self, x, adj):
        return F.relu(self.gc1(x, adj)).detach()


class GCN_homo(nn.Module):
    def __init__(self, nfeat, adj, nhid, out, dropout):
        super(GCN_homo, self).__init__()
        self.gc1 = GraphConvolution_homo(nfeat, adj, nhid)
        self.gc2 = GraphConvolution_homo(nhid, adj, nhid)
        self.gc3 = GraphConvolution_homo(nhid, adj, out)
        self.dropout = dropout


    def forward(self, x, adj, bi_adj, output, labels_for_lp):
        x, y_hat, mask = self.gc1(x, adj, bi_adj, output, labels_for_lp)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training = self.training)
        #x_2 = F.relu(self.gc2(x, adj, bi_adj, output))
        #x_2 = F.dropout(x_2, self.dropout, training=self.training)
        x_3, y_hat, mask = self.gc3(x, adj, bi_adj, output, labels_for_lp)
        #return torch.cat((x, x_2), dim=1)
        return x_3, y_hat, mask


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class MLP(nn.Module):
    def __init__(self, n_feat, n_hid, nclass, dropout):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_feat, n_hid),
            #nn.ReLU(),
            nn.Linear(n_hid, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.mlp(x)

    def get_emb(self, x):
        return self.mlp[0](x).detach()




class HGCN(nn.Module):
    def __init__(self, nfeat, adj, nclass, nhid1, nhid2, n, dropout):
        super(HGCN, self).__init__()
        self.GCN1 = GCN_homo(nfeat, adj, nhid1, nhid2, dropout)
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            #nn.Linear(nhid2, nhid2),
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, adj, bi_adj, output, labels_for_lp):
        emb, y_hat, mask = self.GCN1(x, adj, bi_adj, output, labels_for_lp)
        output = self.MLP(emb)
        return output, F.log_softmax(y_hat, dim=1), mask, emb
