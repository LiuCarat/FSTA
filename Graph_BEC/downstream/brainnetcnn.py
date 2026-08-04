import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectedE2E(nn.Module):
    def __init__(self, in_channels, out_channels, nodes_num):
        super().__init__()
        self.nodes_num = nodes_num
        self.outgoing = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, nodes_num),
        )
        self.incoming = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(nodes_num, 1),
        )

    def forward(self, inputs):
        outgoing = self.outgoing(inputs).expand(-1, -1, -1, self.nodes_num)
        incoming = self.incoming(inputs).expand(-1, -1, self.nodes_num, -1)
        return outgoing + incoming


class DirectedE2N(nn.Module):
    def __init__(self, in_channels, out_channels, nodes_num):
        super().__init__()
        self.outgoing = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, nodes_num),
        )
        self.incoming = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(nodes_num, 1),
        )
        self.fuse = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, inputs):
        outgoing = self.outgoing(inputs)
        incoming = self.incoming(inputs).transpose(2, 3)
        return self.fuse(torch.cat([outgoing, incoming], dim=1))


class DirectedBrainNetCNN(nn.Module):
    def __init__(
        self,
        nodes_num,
        e2e_channels=(4, 8),
        e2n_channels=16,
        n2g_channels=16,
        fc_channels=8,
        dropout=0.5,
    ):
        super().__init__()
        first_e2e, second_e2e = e2e_channels
        self.e2e1 = DirectedE2E(2, first_e2e, nodes_num)
        self.e2e2 = DirectedE2E(first_e2e, second_e2e, nodes_num)
        self.e2n = DirectedE2N(second_e2e, e2n_channels, nodes_num)
        self.n2g = nn.Conv2d(
            e2n_channels,
            n2g_channels,
            kernel_size=(nodes_num, 1),
        )
        self.fc1 = nn.Linear(n2g_channels, fc_channels)
        self.fc2 = nn.Linear(fc_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        output = F.leaky_relu(self.e2e1(inputs), negative_slope=0.1)
        output = F.leaky_relu(self.e2e2(output), negative_slope=0.1)
        output = F.leaky_relu(self.e2n(output), negative_slope=0.1)
        output = F.leaky_relu(self.n2g(output), negative_slope=0.1)
        output = output.flatten(start_dim=1)
        output = self.dropout(F.leaky_relu(self.fc1(output), negative_slope=0.1))
        return self.fc2(output).squeeze(-1)
