import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, dropout=0.5, attention=True):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])
        self.conv3 = GCNConv(hidden_channels[1], hidden_channels[2])
        self.fc = nn.Linear(hidden_channels[2], out_channels)
        self.dropout = dropout
        self.attention = attention
        if attention:
            self.attn = nn.Linear(hidden_channels[2], 1)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        
        if self.attention:
            attn_weights = torch.softmax(self.attn(x), dim=0)
            x = x * attn_weights
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        x = torch.sigmoid(self.fc(x))
        return x, attn_weights if self.attention else None