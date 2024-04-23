import torch 
import os
import torch.nn as nn
def attention_mask(attention_node ,neighborhood_nodes):
    pass

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
        concat = torch.concat(attention_node, dim=1)
        output = self.fc(concat)
        output = self.activation_fun(output)
        output = self.softmax(output)
        return output
    
class GAT_layer(nn.Module):
    def __init__(self, in_feature, out_feature, num_attn, num_nodes, method='concat') -> None:
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.num_attn = num_attn
        attention_mask = Attention(self.in_feature, self.out_feature)
        self.multi_attn = nn.ModuleList([attention_mask for i in range(self.num_attn)])
        self.activation_fun = nn.ModuleList([nn.Sigmoid() for i in range(self.num_attn)])
        self.fc = nn.ModuleList([nn.Linear(self.out_feature, num_nodes) for i in range(self.num_attn)])
        self.fc = nn.Linear(self.out_feature, num_nodes)
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
                
            
    def attention_forward(self, x, previous_hidden, edge_idx, idx=0):
        edge_ij = [previous_hidden, x[edge_idx]]
        e_ij = self.multi_attn[idx](edge_ij)
        hidden_agg = self.fc[idx](e_ij)
        hidden_agg = self.activation_fun[idx](hidden_agg)
        return hidden_agg
    
    def concat(self):
        final_hidden = self.hidden_state[0]
        for m in range(1, self.num_attn):
            final_hidden = torch.concat([final_hidden, self.hidden_state[m]], dim=1)
        return final_hidden
    def sum(self):
        final_hidden = self.hidden_state[0]
        for m in range(1, self.num_attn):
            final_hidden = torch.add(final_hidden, self.hidden_state[m])
            
class GAT(nn.Module):
    def __init__(self, in_features, out_feature, num_layer, num_attn, num_nodes, num_classes) -> None:
        super().__init__()
        self.in_feature = in_features
        self.out_feature = out_feature
        self.num_layers = num_layer
        self.num_classes = num_classes
        self.GAT_init = GAT_layer(self.in_feature, self.out_feature[0], num_attn[0], num_nodes)
        self.GAT_layer = nn.Module([self.GAT_init, GAT_layer(self.out_feature[k-1]*num_attn[k-1], self.out_feature[k], num_attn[k], num_nodes) for k in range(1, num_layer-1)])
        self.GAT_final = GAT_layer(self.in_feature[-1], self.out_feature[-1], num_attn[-1], num_nodes, method='avg')
        self.fc = nn.Linear(self.out_feature, self.num_classes)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, input_features, edg_idx):
        for i in range(self.num_layers - 1):
            input_features = self.GAT_layer[i](input_features, edg_idx)
        output = self.GAT_final(input_features)
        output = self.fc(output)
        output = self.softmax(output)
        return output