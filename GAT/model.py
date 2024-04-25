import scipy.sparse
import torch 
import os
import torch.nn as nn
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torchsummary import summary
from torch_geometric import datasets, data
from torch.utils.data import Dataset
import scipy
import numpy as np
def attention_mask(attention_node ,neighborhood_nodes):
    pass

class CustomeDataloader(Dataset):
    def __init__(self, dataset, indices) -> None:
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        
    def __getitem__(self, index):
        item = {
            'input': self.dataset.x[index],
            'indices': self.indices[index],
            'lables': self.dataset.y[index]
        }
        return item
        
    def __len__(self):
        return (self.dataset.x).shape[0]
# class Attention(nn.Module):
#     def __init__(self, in_features, out_features, adj_matrix, dropout=0.1):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.adj_matrix = adj_matrix
#         self.dropout = dropout
#         self.fc = nn.Linear(self.in_features, self.out_features)
#         # self.fc2 = nn.Linear(self.in_features, self.out_features)
#         self.activation_fun = nn.LeakyReLU()
#         self.softmax = nn.Softmax(dim=-1)
#     def forward(self, node_feature):
#         # concat = torch.concat(attention_node, dim=2)
#         # output = self.fc(node_feature)
#         # output = torch.concat([output1, output2], dim=-1)
#         attention_score = torch.matmul(node_feature, torch.transpose(node_feature))
#         attention_score = self.activation_fun(attention_score)
#         attention_coeff = torch.where(self.adj_matrix > 0, attention_score, 1)
#         output = self.softmax(attention_coeff)
#         return output
        
    
class Attention_layer(nn.Module):
    def __init__(self, in_feature, out_feature, adj_matrix, method='concat') -> None:
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.adj_matrix = adj_matrix
        self.LeakReLU = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(self.in_feature, self.out_feature)
        self.method = method
    
    def forward(self, x):
        x = self.fc(x)
        attention_score = torch.matmul(x, torch.t(x))
        attention_score = self.LeakReLU(attention_score)
        attention_mask = torch.where(self.adj_matrix > 0, attention_score, 1)
        attention_coeff = self.softmax(attention_mask)
        updated_hidden = torch.matmul(attention_coeff, x)
        updated_hidden = self.activation_fun(updated_hidden)
        return updated_hidden
                
        
            
class MultiAttention(nn.Module):
    def __init__(self, in_feature, out_feature, adj_matrix, num_attn:int = 1, method="concat") -> None:
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.adj_matrix = adj_matrix
        self.K = num_attn
        self.attention_layer = nn.ModuleList([Attention_layer(self.in_feature, out_feature, adj_matrix) for i in range(self.num_attn)])
        self.method = method
    def forward(self,x):
        self.output_layer = list()
        for idx_layer in range(self.K):
            output_layer.append(self.attention_layer[idx_layer](x))
        output_layer = torch.Tensor(output_layer)
        output_layer = torch.reshape(output_layer, (1,2,0))
        if self.method == "concat":
            output = torch.flatten(output_layer, dim=-1)
        elif self.method == "mean":
            output = torch.mean(output_layer, dim=-1)
        return output
        
class GAT_layer(nn.Module):
    def __init__(self, in_features, hidden_dim, adj_matrix ,num_attn: int = 1, method="concat") -> None:
        super().__init__()
        self.in_feature = in_features
        self.hidden_dim = hidden_dim
        self.num_attn = num_attn
        self.adj_matrix = adj_matrix
        self.fc1 = nn.Linear(self.in_feature, self.hidden_dim)
        if self.num_attn > 1:
            self.attention_mechanism = MultiAttention(self.in_feature, self.hidden_dim, method)
        else:
            self.attention_mechanism = Attention_layer(self.in_feature, self.hidden_dim)
    def forward(self, input_features):
        output = self.attention_mechanism(input_features)
        output = self.fc(output)
        output = self.softmax(output)
        return output

        
        
class GAT(nn.Module):
    def __init__(self, in_feature, hidden_dim: int| list, out_feature, numclasses, adj_matrix, num_attention) -> None:
        super().__init__()
        self.in_feature = in_feature
        self.hidden_dim = hidden_dim
        self.out_feature = out_feature
        self.classes = numclasses
        self.num_attention = num_attention
        self.adj_matrix = adj_matrix
        
        if isinstance(hidden_dim, int):
            self.layer_init = None
            self.layerGAT = GAT_layer(self.in_feature, self.hidden_dim, self.adj_matrix, self.num_attention)
        else:
            self.layer_init = GAT_layer(self.in_feature, self.hidden_dim[0], self.adj_matrix, self.num_attention)
            self.layerGAT = nn.Sequential(GAT_layer(in_features=self.hidden_dim[i-1]*self.num_attention, 
                                                    hidden_dim=self.hidden_dim, 
                                                    adj_matrix=self.adj_matrix, 
                                                    num_attn=self.num_attention, 
                                                    method="mean") 
                                          for i in range(1, len(self.hidden_dim)))
        
        self.layer_final = GAT_layer(self.hidden_dim[-1]*num_attention, self.out_feature, self.adj_matrix, self.num_attention, method="mean")
        self.fc = nn.Linear(self.out_feature, self.classes)
        self.softmax = nn.Softmax(dim=-1)
def main():
    # gat = GAT(in_features=14, out_feature=[32, 16], num_layer=2, num_attn=[2,2], num_classes=7, edge_idx=(0,1))
    # edge_mask = torch.randint(0,5, (32,1))
    # hidden = torch.rand(32,5,14)
    # print(edge_mask*hidden)
    # print(gat)
    # summary(gat, (32,5,14))
    data = datasets.Planetoid('./datasets', 'Cora')

    dataset = data[0]
    num_node = 2708
    idx_trainmask = dataset.y[dataset.train_mask]
    idx_testmask = [(dataset['x'][i], i) for i in range(2708) if dataset['test_mask'][i] == True ]
    idx_valmask = [(dataset['x'][i], i) for i in range(2708) if dataset['val_mask'][i] == True ]


    # adjacency matrix
    egde_connect_state = torch.ones(10556)
    coo = scipy.sparse.coo_matrix((egde_connect_state, (dataset.edge_index[1], dataset.edge_index[0])), shape=(2708, 2708))
    adj_matrix = coo.toarray() + np.eye(2708)
    

if __name__ == '__main__':
    main()