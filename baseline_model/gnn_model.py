import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs)-1:  # Add ReLU only between layers
                x = F.relu(x)
        return x
        # for conv in self.convs:
        #     x = conv(x, edge_index)
        #     x = F.relu(x) # ovdje mozda ne ak je zadnji sloj
        # return x

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x
        
class EdgeScorer(nn.Module):
    def __init__(self, node_dim, hidden_dim, edge_feat_mode='concat'):
        super().__init__()
        self.mode = edge_feat_mode
        if self.mode == 'dot':
            self.mlp = None
            in_dim = None
        else:
            in_dim = {
                'concat': 2 * node_dim,
                'mean': node_dim,
                'diff': node_dim
                # ,'dot': node_dim
            }[self.mode]
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

    def edge_features(self, x, edge_index):
        row, col = edge_index
        if self.mode == 'concat':
            edge_feat = torch.cat([x[row], x[col]], dim=1)
        elif self.mode == 'mean':
            edge_feat = (x[row] + x[col]) / 2
        elif self.mode == 'diff':
            edge_feat = x[row] - x[col]
        elif self.mode == 'dot':
            # edge_feat = (x[row] * x[col])
            # edge_feat = np.dot(x[row], x[col])
            edge_feat = (x[row] * x[col]).sum(dim=-1)
        else:
            raise ValueError(f"Unknown edge_feat_mode: {self.mode}")
        # print(edge_feat)
        return edge_feat

    def forward(self, x, edge_index):
        edge_feat = self.edge_features(x, edge_index)
        
        if self.mode == 'dot':
            return edge_feat
        
        score = self.mlp(edge_feat).squeeze(-1)
        return score
