import numpy as np
import torch
import random
from torch_geometric.nn import GIN, GCN, GAT, GraphSAGE, GCNConv
from torch_geometric.datasets import Reddit, Planetoid, Amazon, Flickr, Coauthor
from torch_geometric.loader import DataLoader
from torch.nn import MSELoss, CrossEntropyLoss, Linear, ReLU, BatchNorm1d, Dropout, Sequential, Sigmoid
from torch_geometric.transforms import NormalizeFeatures
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
    batch_size = cfg.getint('batch_size')
    
    embedding_layer_num = cfg.getint('embedding_layer_num')
    embedding_hidden_size = cfg.getint('embedding_hidden_size')
    decoder_layer_num = cfg.getint('decoder_layer_num')

    mask_std = cfg.getfloat('mask_std')
    mask_ratio = cfg.getfloat('mask_ratio')

    victim_epochs = cfg.getint('victim_epochs')
    ssl_epochs = cfg.getint('ssl_epochs')
    surrogate_epochs = cfg.getint('surrogate_epochs')

    query_limit = cfg.getint('query_limit')

    tqdm_off = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    for num, (victim_cfg, surrogate_cfg, dataset_name) in enumerate(product(victims, surrogates, datasets), 1):
        
        if dataset_name in ['cora','citeseer','pubmed']:
            dataset = Planetoid(root='./datasets/Planetoid', name=dataset_name)
        elif dataset_name in ['flickr']:
            dataset = Flickr(root='./datasets/Flickr')
        elif dataset_name in ['cs','physics']:
            dataset = Coauthor(root='./datasets/Coauthor', name=dataset_name)
        else:
            raise ValueError(f"Dataset inválido: {dataset_name} não está disponível na implementação")
        
        dataset.to(device)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        victim = victim_cfg['model'] 
        victim_hidden_size = victim_cfg['hidden_size'] 
        victim_layer_num = victim_cfg['layers'] 
        
        surrogate_conv = surrogate_cfg['conv'] 
        surrogate_act = surrogate_cfg['act']
        
        print(f"Experimento #{num} - Modelo Vítima: {victim.upper()} - Modelo Surrogate: {surrogate_conv.upper()} com {surrogate_act} - Dataset: {dataset_name} ")
    
        if victim == 'gat':
            victim_model = GAT(in_channels=dataset.num_features, hidden_channels=victim_hidden_size, out_channels=dataset.num_classes, num_layers=victim_layer_num, act='relu', dropout=0.5).to(device)
        elif victim == 'gcn':
            victim_model = GCN(in_channels=dataset.num_features, hidden_channels=victim_hidden_size, out_channels=dataset.num_classes, num_layers=victim_layer_num, act='relu', dropout=0.5).to(device)
            
        victim_optimizer = torch.optim.Adam(victim_model.parameters(), lr=1e-3)
        
        # treinar o classificador da vitima nos embeddings do dataset rotulado
        for _ in tqdm(range(victim_epochs), desc='Treino Vitima', disable=tqdm_off):
            for batch in loader:
                victim_model.train()

                victim_optimizer.zero_grad()
                
                pred = victim_model(batch.x, batch.edge_index)

                loss = CrossEntropyLoss()(pred[batch.train_mask], batch.y[batch.train_mask])
                
                loss.backward()
                
                victim_optimizer.step()
                
        victim_model.eval()
        
        y_true = []
        y_pred = []

        for batch in loader:      
            pred = victim_model(batch.x, batch.edge_index)
            pred = torch.argmax(pred, dim=-1)
            
            y_pred.append(pred[batch.test_mask].cpu())
            y_true.append(batch.y[batch.test_mask].cpu())
            
        y_pred = torch.cat(y_pred).numpy()
        y_true = torch.cat(y_true).numpy()
                
        print("Acurácia da Vítima:",accuracy_score(y_true, y_pred))
        
        #treinar o embedding do surrogate
        encoder = Encoder(dim_in=dataset.num_features, dim_out=embedding_hidden_size, num_layers=embedding_layer_num, conv=surrogate_conv, act=surrogate_act)
        encoder.to(device)
        
        decoder = DecoderMLP(embedding_hidden_size, dim_out=dataset.num_features, num_layers=decoder_layer_num)
        decoder.to(device)
        
        optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=1e-3)
        
        lagraph_loss = LaGraphNodeLoss()
        ssl_history = []

        for _ in tqdm(range(ssl_epochs), desc='Treino SSL', disable=tqdm_off):
            for batch in loader:
                encoder.train()
                decoder.train()
                
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                
                noisy_node_feature, mask = node_mask(batch.x, batch.train_mask, std=mask_std, ratio=mask_ratio, device=device) 
                
                embedding_original = encoder(batch.x, batch.edge_index)
                embedding_noisy = encoder(noisy_node_feature, batch.edge_index)
                
                reconstructed = decoder(embedding_original)

                loss = lagraph_loss(batch.x[batch.train_mask], reconstructed[batch.train_mask], embedding_original[batch.train_mask], embedding_noisy[batch.train_mask], mask[batch.train_mask]) 
                ssl_history.append(loss.item())
                loss.backward()
                
                optimizer_encoder.step()
                optimizer_decoder.step()
        
        encoder.eval()

        surrogate_head = HeadMLP(embedding_hidden_size, dataset.num_classes).to(device)
        surrogate_optimizer = torch.optim.Adam(surrogate_head.parameters(), lr=1e-2)
        
        # escolher os nós que vao ser usados pelo adversario por k-means no dataset sem rotulos
        selected_idx = select_indices(encoder(dataset[0].x, dataset[0].edge_index).cpu().detach(), query_limit)
        selected_idx = torch.sort(torch.tensor(selected_idx)).values
        
        random_idx = torch.randperm(dataset[0].x.size()[0])[:query_limit].sort().values
        
        # usar a vitima para encontrar os rotulos dos embeddings
        victim_pred = victim_model(dataset[0].x, dataset[0].edge_index)
        victim_pred = torch.argmax(victim_pred, dim=-1)
        
        # treinar o classificador do surrogate nos rotulos obtidos da vitima
        for _ in tqdm(range(surrogate_epochs), desc='Treino Surrogate', disable=tqdm_off):
            for batch in loader:
                surrogate_head.train()

                victim_optimizer.zero_grad()
                
                embedding = encoder(batch.x, batch.edge_index)

                pred = surrogate_head(embedding[selected_idx])
                loss = CrossEntropyLoss()(pred, victim_pred[selected_idx])
                
                loss.backward()
                
                surrogate_optimizer.step()
                
        surrogate_head.eval()
        
        y_true = []
        y_pred = []

        for batch in loader:    
            
            embedding = encoder(batch.x, batch.edge_index)
            
            pred = surrogate_head(embedding[batch.test_mask])
            pred = torch.argmax(pred, dim=-1)
            
            y_pred.append(pred.cpu())
            y_true.append(batch.y[batch.test_mask].cpu())
            
        y_pred = torch.cat(y_pred).numpy()
        y_true = torch.cat(y_true).numpy()
                
        print("Acurácia do Surrogate (Select):",accuracy_score(y_true, y_pred))
        
        # agora vamos utilizar os nós escolhidos randomicamente ao invés dos obtidos pelo k-means
        
        surrogate_head = HeadMLP(embedding_hidden_size, dataset.num_classes).to(device)
        surrogate_optimizer = torch.optim.Adam(surrogate_head.parameters(), lr=1e-2)

        victim_pred = victim_model(dataset[0].x, dataset[0].edge_index)
        victim_pred = torch.argmax(victim_pred, dim=-1)

        for _ in tqdm(range(surrogate_epochs), desc='Treino Surrogate', disable=tqdm_off):
            for batch in loader:
                surrogate_head.train()

                victim_optimizer.zero_grad()
                
                embedding = encoder(batch.x, batch.edge_index)

                pred = surrogate_head(embedding[random_idx])
                loss = CrossEntropyLoss()(pred, victim_pred[random_idx])
                
                loss.backward()
                
                surrogate_optimizer.step()
                
        surrogate_head.eval()
        
        y_true = []
        y_pred = []

        for batch in loader:    
            
            embedding = encoder(batch.x, batch.edge_index)
            
            pred = surrogate_head(embedding[batch.test_mask])
            pred = torch.argmax(pred, dim=-1)
            
            y_pred.append(pred.cpu())
            y_true.append(batch.y[batch.test_mask].cpu())
            
        y_pred = torch.cat(y_pred).numpy()
        y_true = torch.cat(y_true).numpy()
                
        print("Acurácia do Surrogate (Random):",accuracy_score(y_true, y_pred))
    
    
if __name__ == "__main__":
    main()