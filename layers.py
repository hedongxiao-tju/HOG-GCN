import math

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution_homo(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, adj, out_features, bias=True):
        super(GraphConvolution_homo, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_bi = Parameter(torch.FloatTensor(in_features, out_features))
        self.w = Parameter(torch.FloatTensor(1))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.adjacency_mask = Parameter(adj.clone())
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        stdv_bi = 1. / math.sqrt(self.weight_bi.size(1))
        self.weight_bi.data.uniform_(-stdv_bi, stdv_bi)
        self.w.data.uniform_(0.5, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, bi_adj, output, labels_for_lp):

        new_bi = bi_adj.clone()
        new_bi = new_bi * self.adjacency_mask
        new_bi = F.normalize(new_bi, p=1, dim=1)
        identity = torch.eye(adj.shape[0])
        output = output.exp()
        homo_matrix = torch.matmul(output, output.t())
        homo_matrix = 0.4 * homo_matrix + 1 * new_bi
        y_hat = torch.mm(new_bi, labels_for_lp)

        bi_adj = torch.mul(bi_adj, homo_matrix)


        with torch.no_grad():
            bi_row_sum = torch.sum(bi_adj, dim=1, keepdim=True)
            bi_r_inv = torch.pow(bi_row_sum, -1).flatten()  # np.power(rowsum, -1).flatten()
            bi_r_inv[torch.isinf(bi_r_inv)] = 0.
            bi_r_mat_inv = torch.diag(bi_r_inv)
        bi_adj = torch.matmul(bi_r_mat_inv, bi_adj)


        support = torch.mm(input, self.weight)
        support_bi = torch.mm(input, self.weight_bi)
        output = torch.spmm(identity, support)
        output_bi = torch.spmm(bi_adj, support_bi)
        output = output + torch.mul(self.w, output_bi)

        if self.bias is not None:
            return output + self.bias, y_hat, homo_matrix
        else:
            return output, y_hat, homo_matrix

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
