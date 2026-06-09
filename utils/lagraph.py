import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, PReLU, BatchNorm1d, Dropout
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch.nn.functional import relu, prelu, leaky_relu
import numpy as np

class Encoder(torch.nn.Module):
    def __init__(self, dim_in=32, dim_out=32, num_layers=5, act='relu', conv='gcn'):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        
        self.conv_list = torch.nn.ModuleList()
        self.batchnorm_list = torch.nn.ModuleList()
        
        activations = {
            'relu': relu,
            'prelu': prelu,
            'leakyrelu': leaky_relu
        }
        
        conv_layers = {
            'gcn': GCNConv,
            'gat': GATConv,
            'sage': SAGEConv,
            'gin': GINConv
        }
        
        self.conv = conv_layers[conv]  
        self.act = activations[act]
        
        for layer_idx in range(num_layers):
            start_dim = dim_out if layer_idx else dim_in
            if conv == 'gin':
                mlp = torch.nn.Sequential(
                torch.nn.Linear(start_dim, dim_out),
                torch.nn.ReLU()
                )
                self.conv_list.append(self.conv(mlp))
            else:
                self.conv_list.append(self.conv(start_dim, dim_out))
            self.batchnorm_list.append(BatchNorm1d(dim_out))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.conv_list[i](x, edge_index)
            x = self.act(x)
            x = self.batchnorm_list[i](x)

        return x

class DecoderGCN(torch.nn.Module):
    def __init__(self, dim_in=32, dim_out=32, num_layers=5):
        super(DecoderGCN, self).__init__()
        
        layers = nn.Sequential(
            GCNConv(dim_in, dim_in),
            PReLU(),
            BatchNorm1d(dim_in)
        )
        
        self.network = nn.Sequential(
            *[layers for i in range(num_layers-1)],
            GCNConv(dim_in, dim_out),
            PReLU()
        )

    def forward(self, x):
        return self.network(x)


class DecoderMLP(torch.nn.Module):
    def __init__(self, dim_in=32, dim_out=32, num_layers=5, dropout=0.5):
        super(DecoderMLP, self).__init__()
        
        layers = nn.Sequential(
            Linear(dim_in, dim_in),
            BatchNorm1d(dim_in),
            ReLU(),
            Dropout(dropout)
        )
        
        self.fc = nn.Sequential(
            *[layers for i in range(num_layers-1)],
            Linear(dim_in, dim_out),
            ReLU()
        )

    def forward(self, x):
        return self.fc(x)
    
class LaGraphNodeLoss(torch.nn.Module):
    '''
    _Node-level loss_ como definido pelo artigo [Self-Supervised Representation Learning via Latent Graph Prediction](https://arxiv.org/pdf/2202.08333)
    '''
    
    def __init__(self):
        super(LaGraphNodeLoss, self).__init__()

    def forward(self, original, reconstructed, representation_orig, representation_rec, mask, alpha=2):
        loss_reconstruction = nn.MSELoss()(original, reconstructed)
        loss_invariance = nn.MSELoss(reduction='none')(representation_orig, representation_rec)
        loss_invariance = torch.sqrt(torch.sum(loss_invariance * mask[:,None]) / torch.sum(mask).clamp(min=1))
        loss = loss_reconstruction + alpha * loss_invariance
        
        return loss
    
def node_mask(x, idx_mask, std, ratio, device='cpu'):
    '''
    Recebe um mini-batch de grafos (formato PyG) e retorna o grafo com uma chance de ``ratio`` individual de um nó ser substituído por ruído com média 0 e desvio padrão ``std``.
    
    Parameters
    ----------
    x : ndarray da forma (n_node, n_feat)
        Mini-batch de grafos
    std: float
        Desvio padrão do ruído
    ratio: float
        Chance de um nó ser substituído por ruído

    Returns
    -------
    x : ndarray da forma (n_node, n_feat)
        Mini-batch com ruído inserido
    mask : ndarray da forma (n_node,)
        Máscaras com nós ruidosos para cada grafo
    '''
    
    x_masked = x.clone()

    node_num, feat_dim = x.size()

    idx = torch.where(idx_mask)[0]

    mask_size = int(ratio * len(idx))

    perm = torch.randperm(len(idx), device=x.device)
    mask_idx = idx[perm[:mask_size]]

    noise = torch.normal(
        mean=0.0,
        std=std,
        size=(node_num, feat_dim),
        device=x.device
    )

    mask = torch.zeros(node_num, dtype=torch.bool, device=x.device)
    mask[mask_idx] = True

    x_masked[mask] = noise[mask]

    return x_masked, mask
    
    x_masked = x.clone()

    node_num, feat_dim = x[idx_mask].size()
    mask_size = int(ratio * node_num)

    mask_idx = torch.randperm(node_num, device=device)[:mask_size]

    noise = torch.normal(
        mean=0.0,
        std=std,
        size=(node_num, feat_dim),
        device=device
    )

    mask = torch.zeros(node_num, dtype=torch.bool, device=device)
    mask[mask_idx] = 1

    x_masked[mask] = noise[mask]

    return x_masked, mask