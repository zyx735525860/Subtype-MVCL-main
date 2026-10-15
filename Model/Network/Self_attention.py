import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(SelfAttentionEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # 线性层用于计算查询、键和值
        self.query_layer = nn.Linear(input_dim, hidden_dim)
        self.key_layer = nn.Linear(input_dim, hidden_dim)
        self.value_layer = nn.Linear(input_dim, hidden_dim)

        # 线性层用于输出
        self.output_layer = nn.Linear(hidden_dim, input_dim)

        # 层归一化
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, input_data):
        # 计算查询、键和值
        queries = self.query_layer(input_data)
        keys = self.key_layer(input_data)
        values = self.value_layer(input_data)

        # 计算注意力权重
        attention_weights = torch.matmul(queries, keys.transpose(-1, -2)) / (self.hidden_dim ** 0.5)
        attention_weights = F.dropout(F.softmax(attention_weights, dim=-1), p=self.dropout, training=self.training)

        # 计算加权和
        weighted_values = torch.matmul(attention_weights, values)

        # 输出层
        output = F.dropout(self.output_layer(weighted_values), p=self.dropout, training=self.training)

        # 层归一化
        output = self.layer_norm(output + input_data)

        return output
