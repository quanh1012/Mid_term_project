import torch 
import os
import torch.nn as nn
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torchsummary import summary
from torch_geometric import datasets, data
from torch.utils.data import Dataset
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
class Attention(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.fc = nn.Linear(2*self.in_features, self.out_features)
        self.activation_fun = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=2)
    def forward(self, attention_node):
        concat = torch.concat(attention_node, dim=2)
        output = self.fc(concat)
        output = self.activation_fun(output)
        output = self.softmax(output)
        return output
    
class GAT_layer(nn.Module):
    def __init__(self, in_feature, out_feature, num_attn, method='concat') -> None:
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.num_attn = num_attn
        attention_mask = Attention(self.in_feature, self.out_feature) #28x32
        self.multi_attn = nn.ModuleList([attention_mask for i in range(self.num_attn)])
        self.activation_fun = nn.ModuleList([nn.Sigmoid() for i in range(self.num_attn)])
        self.fc = nn.ModuleList([nn.Linear(self.in_feature, self.out_feature) for i in range(self.num_attn)])
        # self.fc = nn.Linear(self.out_feature, num_nodes)
        self.method = method
        # self.activation_fun = nn.Sigmoid()
    
    def forward(self, x, edge_idx):
        hidden_past = x
        if self.num_attn == 1:
            return self.attention_forward(x, hidden_past, edge_idx)
        else:
            self.hidden_state = list()
            for i in range(self.num_attn):
                hidden_update = self.attention_forward(x, hidden_past, edge_idx, i)
                self.hidden_state.append()
                hidden_past = hidden_update
            if self.method == 'concat':
                return self.concat()
            else:
                return self.sum()
                
            
    def attention_forward(self, x, edge_mask, idx=0):
        previous_hidden = torch.mm()
        edge_ij = [previous_hidden, x]
        print(edge_ij.shape)
        e_ij = self.multi_attn[idx](edge_ij) 
        hidden_agg = self.fc[idx](e_ij)
        hidden_agg = self.activation_fun[idx](hidden_agg)
        return hidden_agg
    
    def concat(self):
        final_hidden = self.hidden_state[0]
        for m in range(1, self.num_attn):
            final_hidden = torch.concat([final_hidden, self.hidden_state[m]], dim=2)
        return final_hidden
    def sum(self):
        final_hidden = self.hidden_state[0]
        for m in range(1, self.num_attn):
            final_hidden = torch.add(final_hidden, self.hidden_state[m])
            
class GAT(nn.Module):
    def __init__(self, in_features, hidden_layer, out_feature, num_layer, num_attn, num_classes, edge_idx) -> None:
        super().__init__()
        self.in_feature = in_features
        self.out_feature = out_feature
        self.num_layers = num_layer
        self.num_classes = num_classes
        self.edge_idx = edge_idx
        self.GAT_init = GAT_layer(self.in_feature, self.out_feature[0], num_attn[0])
        if self.num_layers > 2:
            self.GAT_layer = nn.Module([GAT_layer(self.out_feature[k-1]*num_attn[k-1], self.out_feature[k], num_attn[k]) for k in range(1, num_layer-1)])
        self.GAT_final = GAT_layer(self.out_feature[-2]*num_attn[-1], self.out_feature[-1], num_attn[-1], method='avg')
        self.fc = nn.Linear(self.out_feature[-1], self.num_classes)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, input_features):
        input_features = self.GAT_init(input_features, self.edge_idx)
        if self.num_layers > 2: 
            for i in range(self.num_layers - 1):
                input_features = self.GAT_layer[i](input_features, self.edge_idx)
        output = self.GAT_final(input_features)
        output = self.fc(output)
        output = self.softmax(output)
        return output

        
        
def main():
    # gat = GAT(in_features=14, out_feature=[32, 16], num_layer=2, num_attn=[2,2], num_classes=7, edge_idx=(0,1))
    # edge_mask = torch.randint(0,5, (32,1))
    # hidden = torch.rand(32,5,14)
    # print(edge_mask*hidden)
    # print(gat)
    # summary(gat, (32,5,14))
    data = datasets.Planetoid('./datasets', 'Cora')
    # for i, key in enumerate(data[0]):
    #     print(key, data[0][key].shape)
    dataset = data[0]
    num_node = 2708
    idx_trainmask = dataset.y[dataset.train_mask]
    idx_testmask = [(dataset['x'][i], i) for i in range(2708) if dataset['test_mask'][i] == True ]
    idx_valmask = [(dataset['x'][i], i) for i in range(2708) if dataset['val_mask'][i] == True ]
    edge_ij = torch.flip(dataset.edge_index, dims=(0,)).to_sparse(layout=torch.sparse_coo)
    edge_idx = (edge_ij.indices, edge_ij.values)
if __name__ == '__main__':
    main()