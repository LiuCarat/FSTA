import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric as tg

from layers import *

class MLP_header(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, n_classes):
        super(MLP_header, self).__init__()
        self.g1 = nn.Linear(dim_in, dim_hid)
        self.g2 = nn.Linear(dim_hid, dim_out)

        self.f = nn.Linear(dim_out, n_classes)

    def forward(self, x):
        x = F.leaky_relu(self.g1(x))
        x = F.leaky_relu(self.g2(x))
        x = F.normalize(x, p=2, dim=1)
        o = self.f(x)
        return F.log_softmax(o, dim=1), x


class VariDim_Projection(nn.Module):
    def __init__(self, modal_dims, dim_hid, dim_out, n_classes):
        super(VariDim_Projection, self).__init__()

        self.modal_dims = modal_dims
        self.dim_hid = dim_hid
        self.dim_out = dim_out
        self.n_classes = n_classes

        self.Gens = nn.ModuleList()
        for i, dim in enumerate(self.modal_dims):
            self.Gen = MLP_header(dim_in=dim, dim_hid=dim_hid, dim_out=dim_out, n_classes=n_classes)
            self.Gens.append(self.Gen)
            self.add_module('generator_%d_%d' % (dim, i), self.Gen)

    def forward(self, x):
        temp_dim = 0
        bs = x.size(0)
        modal_num = len(self.modal_dims)
        pred_list = []
        proj_list = []
        for i in range(modal_num):
            gen = self.Gens[i]
            data = x[:, temp_dim : temp_dim + self.modal_dims[i]]
            temp_dim += self.modal_dims[i]
            pred, proj = gen(data)
            pred_list.append(pred)
            proj_list.append(proj)
        
        return proj_list, pred_list

class Discriminator(nn.Module):
    def __init__(self, d, d_hid):
        super(Discriminator, self).__init__()
        self.d = d
        #----------------------------#
        # set the number of each layer of the domain discriminator
        self.h_d_1 = d_hid
        self.d1 = nn.Linear(self.d, self.h_d_1)
        self.d2 = nn.Linear(self.h_d_1, 2)
    
    def forward(self, x):
        self.d_logits_1 = F.relu(self.d1(x))
        self.d_logits_2 = self.d2(self.d_logits_1)
        return self.d_logits_2

class GraphLearn(nn.Module):
    def __init__(self, mode, d, th):
        super(GraphLearn, self).__init__()
        self.mode = mode
        self.w = nn.Linear(d, 1)
        self.t = nn.Parameter(torch.ones(1))
        self.p = nn.Linear(d, d)
        self.threshold = nn.Parameter(torch.zeros(1))
        self.th = th
    
    def forward(self, x):
        initial_x = x.clone()
        num, feat_dim = x.size(0), x.size(1)
        
        if self.mode == "Sigmoid-like":
            x = x.repeat_interleave(num, dim = 0)
            x = x.view(num, num, feat_dim)
            diff = abs(x - initial_x)
            diff = diff.pow(2).sum(dim=2).pow(1/2)
            diff = (diff + self.threshold) * self.t
            output = 1 - torch.sigmoid(diff)
            
        elif self.mode == "adaptive-learning":
            # print("x shape:", x.shape)
            x = x.repeat_interleave(num, dim = 0)
            x = x.view(num, num, feat_dim)
            diff = abs(x - initial_x)
            diff = F.relu(self.w(diff)).view(num, num)
            output = F.softmax(diff, dim = 1)
            # print("output shape:", output.shape)
        
        elif self.mode == 'weighted-cosine':
            # print("x shape:", x.shape)
            th = self.th
            x_norm = F.normalize(x,dim=-1)
            score = torch.matmul(x_norm, x_norm.T)
            mask = (score > th).detach().float()
            markoff_value = 0
            output = score * mask + markoff_value * (1 - mask)
            # print("output shape:", output.shape)
        return output

class VariModal_GraphLearn(nn.Module):
    def __init__(self, mode, modal_dims, d, th):
        super(VariModal_GraphLearn, self).__init__()
        self.modal_dims = modal_dims
        self.GCs = nn.ModuleList()
        for i, dim in enumerate(self.modal_dims):
            self.gc = GraphLearn(mode, d, th)
            self.GCs.append(self.gc)
            self.add_module('GraphLearn_%d_%d' % (dim, i), self.gc)
        # self.intra_w = nn.Parameter(torch.tensor([1/len(self.modal_dims)]*len(self.modal_dims)))
        self.weight = nn.Parameter(torch.ones(len(self.modal_dims)), requires_grad=True)
        

    def forward(self, x):
        modal_num = len(self.modal_dims)
        adj_list = []
        for i in range(modal_num):
            gc = self.GCs[i]
            proj = x[i]
            adj = gc(proj)
            adj_list.append(adj)
        stack_adj = torch.stack(adj_list).permute(1, 2, 0)
        # print(stack_adj.shape)
        # comb_adj = torch.matmul(stack_adj, self.intra_w)
        # comb_adj = F.relu(F.linear(stack_adj, self.intra_w.t()))
        # print(comb_adj.shape)
        # comb_adj = adj_list[0]
        self.intra_w = F.softmax(self.weight, 0)
        comb_adj = torch.matmul(stack_adj, self.intra_w)
        return comb_adj

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
            
    def forward(self, input, adj):
        support = torch.mm(input, self.W)
        output = torch.mm(adj, support)
        return output

class GCN(nn.Module):
    def __init__(self, d, dropout, n_classes):
        super(GCN, self).__init__()
        nfeat = d
        nhid = d // 2
        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, n_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x2 = F.dropout(x1, self.dropout, training=self.training)
        x3 = self.gc2(x2, adj)
        return F.log_softmax(x3, dim=1), x2


class SSGC(nn.Module):
    def __init__(self, num_features, nhid, num_classes, K, dropout):
        super(SSGC, self).__init__()
        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.K = K
        self.dropout_ratio = dropout

        self.conv1 = SSGConv(self.num_features, self.nhid, K=self.K, cached=False)
        # self.conv2 = SSGConv(self.nhid, self.nhid, K=self.K, cached=False)

        self.lin1 = nn.Linear(self.nhid, self.nhid)
        self.lin2 = nn.Linear(self.nhid, self.num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(self.lin1(x))
        # x = self.conv2(x, edge_index, edge_weight)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1), x
