import torch
import torch.nn as nn
import torch.nn.functional as F

class Star_net(nn.Module):
    # 模块初始化
    def __init__(self, input_dim, output_dim, star_mutl_dim, act_fun='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.star_mutl_dim = star_mutl_dim
        self.mapping_dim = self.star_mutl_dim * self.output_dim

        self.enc = nn.Linear(self.input_dim, self.output_dim)
        self.fc_1 = nn.Linear(self.output_dim, self.mapping_dim)
        self.fc_2 = nn.Linear(self.output_dim, self.mapping_dim)
        # nn.init.xavier_normal_(self.fc_2.weight)
        self.act_sco = nn.Softmax()
        self.fc_3 = nn.Linear(self.mapping_dim, self.output_dim)

        self.act_fun = act_fun
        if self.act_fun == "gelu":
            self.act = nn.GELU()
        elif self.act_fun == "tahn":
            self.act = nn.Tanh()
        elif self.act_fun == "leakyrelu":
            self.act = nn.LeakyReLU()
        elif self.act_fun == "relu":
            self.act = nn.ReLU()

        # if self.act_fun == 'leakyrelu':
        #     nn.init.kaiming_normal_(self.fc_3.weight, nonlinearity='leaky_relu')  # leaky_relu
        # elif self.act_fun == 'relu':
        #     nn.init.kaiming_normal_(self.fc_3.weight, nonlinearity='relu')
        # elif self.act_fun == 'tahn':
        #     nn.init.xavier_normal_(self.fc_3.weight)
        # if self.fc_3.bias is not None:
        #     nn.init.zeros_(self.fc_3.bias)

        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.norm = nn.BatchNorm1d(self.output_dim)

    # 模块前向传播
    def forward(self, data):
        data = self.enc(data)
        fea = self.fc_1(data)
        fea_sco = self.fc_2(data)
        fea_sco = self.act_sco(fea_sco)
        fea = fea * fea_sco
        fea = self.fc_3(fea)

        fea = self.norm(fea)
        fea = self.act(fea)
        fea = data + fea

        return fea
