import numpy as np
import torch
import random
from torch_geometric.nn import GIN, GCN, GAT, GraphSAGE, GCNConv
from torch_geometric.datasets import Reddit, Planetoid, Amazon, Flickr, Coauthor, WikiCS
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
from torch.nn import MSELoss, CrossEntropyLoss, Linear, ReLU, BatchNorm1d, Dropout, Sequential, Sigmoid
from torch_geometric.transforms import NormalizeFeatures, RandomNodeSplit
import torch.nn.functional as F
from tqdm import tqdm
from utils.lagraph import LaGraphNodeLoss, DecoderGCN, DecoderMLP, node_mask, Encoder
import matplotlib.pyplot as plt
import matplotlib as mlp
from sklearn.metrics import classification_report, pairwise_distances_argmin_min, accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import argparse
import configparser
import json 
from itertools import product

def select_indices(X, k):
    X_norm = StandardScaler().fit_transform(X)

    kmeans = KMeans(
        n_clusters=k,
        n_init=5
    ).fit(X_norm)

    closest, _ = pairwise_distances_argmin_min(
        kmeans.cluster_centers_,
        X_norm
    )

    return closest

class HeadMLP(torch.nn.Module):
    def __init__(self, dim_in=32, dim_out=32):
        super(HeadMLP, self).__init__()
        
        self.fc = Sequential(
            Linear(dim_in, dim_out),
            #Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

def main():
    # adicional: implementar watermark https://arxiv.org/pdf/2110.11024 TBD
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        help='Path para arquivo de config'
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.cfg)

    cfg = config['experimento']
    
    datasets = [x.strip() for x in cfg.get('datasets').split(',')]   
    
    victims = json.loads(cfg.get('victims')) 
    surrogates = json.loads(cfg.get('surrogates'))

    seed = cfg.getint('seed')

    victim_epochs = cfg.getint('victim_epochs')
    surrogate_epochs = cfg.getint('surrogate_epochs')

    query_limit = cfg.getint('query_limit')

    tqdm_off = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    for num, (victim_cfg, surrogate_cfg, dataset_name) in enumerate(product(victims, surrogates, datasets), 1):
        
        if dataset_name in ['photo','computers']:
            dataset = Amazon(root='./datasets/Amazon', name=dataset_name, transform=RandomNodeSplit()).to(device)
            dataset[0].to(device)
            train_mask, test_mask, val_mask = dataset[0].train_mask.to(device), dataset[0].test_mask.to(device), dataset[0].val_mask.to(device)
            train_features, val_features, test_features = dataset.x[train_mask], dataset.x[val_mask], dataset.x[test_mask]
            train_edge_index, _ = subgraph(train_mask, dataset.edge_index, relabel_nodes=True)
            test_edge_index, _ = subgraph(test_mask, dataset.edge_index, relabel_nodes=True)
            val_edge_index, _ = subgraph(val_mask, dataset.edge_index, relabel_nodes=True)
        elif dataset_name in ['cs','physics']:
            dataset = Coauthor(root='./datasets/Coauthor', name=dataset_name, transform=RandomNodeSplit()).to(device)
            train_mask, test_mask, val_mask = dataset[0].train_mask.to(device), dataset[0].test_mask.to(device), dataset[0].val_mask.to(device)
            train_features, val_features, test_features = dataset.x[train_mask], dataset.x[val_mask], dataset.x[test_mask]
            train_edge_index, _ = subgraph(train_mask, dataset.edge_index, relabel_nodes=True)
            test_edge_index, _ = subgraph(test_mask, dataset.edge_index, relabel_nodes=True)
            val_edge_index, _ = subgraph(val_mask, dataset.edge_index, relabel_nodes=True)
        elif dataset_name in ['reddit']:
            dataset = Reddit(root='./datasets/Reddit', name=dataset_name).to(device)
            train_mask, test_mask, val_mask = dataset.train_mask, dataset.test_mask, dataset.val_mask
            train_features, val_features, test_features = dataset.x[train_mask], dataset.x[val_mask], dataset.x[test_mask]
            train_edge_index, _ = subgraph(train_mask, dataset.edge_index, relabel_nodes=True)
            test_edge_index, _ = subgraph(test_mask, dataset.edge_index, relabel_nodes=True)
            val_edge_index, _ = subgraph(val_mask, dataset.edge_index, relabel_nodes=True)
        elif dataset_name in ['wikics']:
            dataset = WikiCS(root='./datasets/WikiCS', is_undirected=True).to(device)
            split = 0
            train_mask, test_mask, val_mask = dataset.train_mask[:,split], dataset.test_mask, dataset.val_mask[:,split]
            train_features, val_features, test_features = dataset.x[train_mask], dataset.x[val_mask], dataset.x[test_mask]
            train_edge_index, _ = subgraph(train_mask, dataset.edge_index, relabel_nodes=True)
            test_edge_index, _ = subgraph(test_mask, dataset.edge_index, relabel_nodes=True)
            val_edge_index, _ = subgraph(val_mask, dataset.edge_index, relabel_nodes=True)
        else:
            raise ValueError(f"Dataset inválido: {dataset_name} não está disponível na implementação")
        
        victim = victim_cfg['model'] 
        victim_hidden_size = victim_cfg['hidden_size'] 
        victim_layer_num = victim_cfg['layers'] 
        
        surrogate_conv = surrogate_cfg['conv'] 
        surrogate_act = surrogate_cfg['act']
        surrogate_hidden_size = surrogate_cfg['hidden_size']
        surrogate_layer_num = surrogate_cfg['layer_num']
        
        print(f"Experimento #{num} - Modelo Vítima: {victim.upper()} - Modelo Surrogate: {surrogate_conv.upper()} com {surrogate_act} - Dataset: {dataset_name} ")
    
        if victim == 'gat':
            victim_model = GAT(in_channels=dataset.num_features, hidden_channels=victim_hidden_size, out_channels=dataset.num_classes, num_layers=victim_layer_num, act='relu', dropout=0.5, heads=4).to(device)
            victim_hidden_size = 4*victim_hidden_size
        elif victim == 'sage':
            victim_model = GraphSAGE(in_channels=dataset.num_features, hidden_channels=victim_hidden_size, out_channels=dataset.num_classes, num_layers=victim_layer_num, act='relu', dropout=0.5).to(device)
            
        victim_optimizer = torch.optim.Adam(victim_model.parameters(), lr=1e-3)
        
        # treinar o classificador da vitima nos embeddings do dataset rotulado
        for _ in tqdm(range(victim_epochs), desc='Treino Vitima', disable=tqdm_off):
            victim_model.train()

            victim_optimizer.zero_grad()
            
            pred = victim_model(train_features, train_edge_index)

            loss = CrossEntropyLoss()(pred, dataset.y[train_mask])
            
            loss.backward()
            
            victim_optimizer.step()
                
        victim_model.eval()
        
        y_true = []
        y_pred = []
   
        pred = victim_model(test_features, test_edge_index)
        pred = torch.argmax(pred, dim=-1)
        
        y_pred.append(pred.cpu())
        y_true.append(dataset.y[test_mask].cpu())
            
        y_pred = torch.cat(y_pred).numpy()
        y_true = torch.cat(y_true).numpy()
                
        print("Acurácia da Vítima:",accuracy_score(y_true, y_pred))
        
        encoder = Encoder(dim_in=dataset.num_features, dim_out=surrogate_hidden_size, num_layers=surrogate_layer_num, conv=surrogate_conv, act=surrogate_act)
        encoder.to(device)
        
        encoder.eval()

        surrogate_head = HeadMLP(surrogate_hidden_size, dataset.num_classes).to(device)
        surrogate_optimizer = torch.optim.Adam(surrogate_head.parameters(), lr=1e-2)
        
        selected_idx = select_indices(encoder(train_features, train_edge_index).cpu().detach(), query_limit)
        selected_idx = torch.sort(torch.tensor(selected_idx)).values
        
        random_idx = torch.randperm(train_features.size()[0])[:query_limit].sort().values
        
        # usar a vitima para encontrar os rotulos dos embeddings
        victim_pred = victim_model(train_features, train_edge_index)
        victim_pred = torch.argmax(victim_pred, dim=-1)
        
        # treinar o classificador do surrogate nos rotulos obtidos da vitima
        for _ in tqdm(range(surrogate_epochs), desc='Treino Surrogate', disable=tqdm_off):
            surrogate_head.train()

            surrogate_optimizer.zero_grad()
            
            embedding = encoder(train_features, train_edge_index)

            pred = surrogate_head(embedding[selected_idx])
            loss = CrossEntropyLoss()(pred, victim_pred[selected_idx])
            
            loss.backward()
            
            surrogate_optimizer.step()
                
        surrogate_head.eval()
        
        y_true = []
        y_pred = []

            
        embedding = encoder(test_features, test_edge_index)
        
        pred = surrogate_head(embedding)
        pred = torch.argmax(pred, dim=-1)
        
        y_pred.append(pred.cpu())
        y_true.append(dataset.y[test_mask].cpu())
            
        y_pred = torch.cat(y_pred).numpy()
        y_true = torch.cat(y_true).numpy()
                
        print("Acurácia do Surrogate (Select):",accuracy_score(y_true, y_pred))
        
        # agora vamos utilizar os nós escolhidos randomicamente ao invés dos obtidos pelo k-means
        
        surrogate_head = HeadMLP(surrogate_hidden_size, dataset.num_classes).to(device)
        surrogate_optimizer = torch.optim.Adam(surrogate_head.parameters(), lr=1e-2)

        victim_pred = victim_model(train_features, train_edge_index)
        victim_pred = torch.argmax(victim_pred, dim=-1)

        for _ in tqdm(range(surrogate_epochs), desc='Treino Surrogate', disable=tqdm_off):
            surrogate_head.train()

            surrogate_optimizer.zero_grad()
            
            embedding = encoder(train_features, train_edge_index)

            pred = surrogate_head(embedding[random_idx])
            loss = CrossEntropyLoss()(pred, victim_pred[random_idx])
            
            loss.backward()
            
            surrogate_optimizer.step()
                
        surrogate_head.eval()
        
        y_true = []
        y_pred = []
            
        embedding = encoder(test_features, test_edge_index)
        
        pred = surrogate_head(embedding)
        pred = torch.argmax(pred, dim=-1)
        
        y_pred.append(pred.cpu())
        y_true.append(dataset.y[test_mask].cpu())
            
        y_pred = torch.cat(y_pred).numpy()
        y_true = torch.cat(y_true).numpy()
                
        print("Acurácia do Surrogate (Random):",accuracy_score(y_true, y_pred))
    
    
if __name__ == "__main__":
    main()