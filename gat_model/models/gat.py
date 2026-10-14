import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch import nn

class GATNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads):
        super().__init__()

        self.gat1 = GATConv(
            in_dim, hidden_dim,
            heads=heads,
            concat=True
        )

        self.gat2 = GATConv(
            hidden_dim * heads,
            out_dim,
            heads=1,
            concat=False
        )

        self.regressor = nn.Linear(out_dim, 1)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = self.regressor(x)
        return x.squeeze()