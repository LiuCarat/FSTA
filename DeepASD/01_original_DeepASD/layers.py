import math
from typing import Optional, Callable, Union

import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset

from torch_geometric.typing import Adj, OptTensor, OptPairTensor, Size
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class SGConv(MessagePassing):
    _cached_x: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SGConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._cached_x = None

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self._cached_x = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)

            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   size=None)

                if self.cached:
                    self._cached_x = x
        else:
            x = cache.detach()

        return self.lin(x)
        # return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)


class SSGConv(MessagePassing):
    _cached_x: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, dropout: float = 0.05, **kwargs):
        super(SSGConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.dropout = dropout

        self._cached_x = None

        self.lin = Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self._cached_x = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)

            alpha = 0.05
            output = alpha * x

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   size=None)

                output = (output + (1 - alpha) * (1. / self.K) * x)

            x = output
            if self.cached:
                self._cached_x = x
        else:
            x = cache.detach()

        return self.lin(x)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)


class CTGConv(MessagePassing):
    _cached_x: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 alpha: float = 0.1, aggr_type: str = 'sgc', norm_type: str = 'arctan',
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, dropout: float = 0.05, **kwargs):
        super(CTGConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K

        self.alpha = alpha
        self.aggr_type = aggr_type
        self.norm_type = norm_type

        self.cached = cached
        self.add_self_loops = add_self_loops
        self.dropout = dropout

        self._cached_x = None

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self._cached_x = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)

            output = self.alpha * x

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   size=None)
                if self.aggr_type == 'ssgc':
                    output = output + (1 - self.alpha) * (1. / self.K) * x
                elif self.aggr_type == 'ssgc_no_avg':
                    output = output + (1 - self.alpha) * x

            if self.aggr_type == 'sgc':
                output = output + (1 - self.alpha) * x

            x = output

            if self.cached:
                self._cached_x = x
        else:
            x = cache.detach()

        if self.norm_type == 'arctan':
            x = 2 * torch.arctan(x) / math.pi
        elif self.norm_type == 'f2':
            x = F.normalize(x, p=2, dim=1)

        return self.lin(x)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, K={}, alpha={}, aggr={}, norm={})'.format(self.__class__.__name__,
                                                                     self.in_channels, self.out_channels,
                                                                     self.K, self.alpha, self.aggr_type, self.norm_type)


class GINConvWeight(MessagePassing):

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINConvWeight, self).__init__(**kwargs)
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        if hasattr(self.nn[0], 'in_features'):
            in_channels = self.nn[0].in_features
        else:
            in_channels = self.nn[0].in_channels
        self.lin = Linear(1, in_channels)

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(edge_index, Tensor):
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = edge_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    self.improved, self.add_self_loops)
                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = cache[0], cache[1]

        elif isinstance(edge_index, SparseTensor):
            cache = self._cached_adj_t
            if cache is None:
                edge_index = edge_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    self.improved, self.add_self_loops)
                if self.cached:
                    self._cached_adj_t = edge_index
            else:
                edge_index = cache

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        single = torch.ones_like(edge_weight)
        if edge_weight is None:
            return x_j
        elif torch.equal(edge_weight, single):
            return edge_weight.view(-1, 1) * x_j
        else:
            # return ((1. + edge_weight.view(-1, 1)) * x_j).relu()
            return ((1. * edge_weight.view(-1, 1)) * x_j).relu()

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


def edge_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
              add_self_loops=True, dtype=None):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        # if add_self_loops:
        #     adj_t = fill_diag(adj_t, fill_value)
        # deg = sparsesum(adj_t, dim=1)
        # deg_inv_sqrt = deg.pow_(-0.5)
        # deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        # adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        # adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        # if add_self_loops:
        #     edge_index, tmp_edge_weight = add_remaining_self_loops(
        #         edge_index, edge_weight, fill_value, num_nodes)
        #     assert tmp_edge_weight is not None
        #     edge_weight = tmp_edge_weight

        # row, col = edge_index[0], edge_index[1]
        # deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        # deg_inv_sqrt = deg.pow_(-0.5)
        # deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        # return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return edge_index, edge_weight
