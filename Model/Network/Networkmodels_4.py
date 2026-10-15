import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE

from Model.Network.Star_block import Star_net

class network(nn.Module):
    # 模型初始化
    def __init__(self, args):
        super(network, self).__init__()
        self.num_view = args.num_view  # 视图数量
        self.hidden_dim = args.hidden_dim  # MLP隐藏维度
        self.dropout = args.dropout  # 丢弃率
        self.head_num = args.head_num  # 自注意力编码器头数
        self.input_dim = args.dim_list  # 多组学输入数据维度
        self.noise_share_unique_dim = args.noise_share_unique_dim  # 噪音、共享、独有部分维度
        self.mu_logvar_dim = args.mu_logvar_dim  # VAE的均值和方差维度
        self.star_mutl_dim = args.star_mutl_dim  # star网络的映射倍数
        self.dec_input_dim = self.mu_logvar_dim + self.noise_share_unique_dim * 2  # 解码器输入维度
        self.unique_integration_dim = self.num_view * self.noise_share_unique_dim  # 多组学独有部分拼接维度
        self.device = args.device  # 运行设备
        self.cluster_number = args.cluster_number  # 聚类数量
        self.mu_logvar_bn = nn.BatchNorm1d(args.mu_logvar_dim)  # 批处理
        self.unique_to_cluster_dim = self.noise_share_unique_dim * 2  # 整合后的独有部分维度
        self.cluster_dim = self.noise_share_unique_dim + self.unique_to_cluster_dim  # 聚类维度
        self.cluster_layer = nn.Parameter(torch.randn(self.cluster_number, self.cluster_dim, dtype=torch.float32))  # 初始化聚类中心
        self.act_fun = args.act  # 激活函数
        if self.act_fun == "gelu":
            self.act = nn.GELU()
        elif self.act_fun == "tahn":
            self.act = nn.Tanh()
        elif self.act_fun == "leakyrelu":
            self.act = nn.LeakyReLU()
        elif self.act_fun == "relu":
            self.act = nn.ReLU()

        self.dropout_layer = nn.Dropout(p=self.dropout)
        # 编码器部分
        # 第一层MLP
        self.encoder_MLP_1 = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.input_dim[view], self.hidden_dim[0]),
                self.act,
                # self.dropout_layer
            ) for view in range(self.num_view)
        )

        # 第二层MLP
        self.encoder_MLP_2 = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.hidden_dim[0], self.hidden_dim[1]),
                self.act,
                self.dropout_layer
            ) for _ in range(self.num_view)
        )

        # 第三层noise_star
        self.noise_star = nn.ModuleList(
            Star_net(self.hidden_dim[-1], self.noise_share_unique_dim, self.star_mutl_dim) for _ in range(self.num_view)
        )

        #第四层share_star
        self.share_star = Star_net(self.hidden_dim[-1], self.noise_share_unique_dim, self.star_mutl_dim)

        # 第五层unique_star
        self.unique_star = nn.ModuleList(
            Star_net(self.hidden_dim[-1], self.noise_share_unique_dim, self.star_mutl_dim) for _ in range(self.num_view)
        )

        # 第六层noise_to_mu
        self.noise_to_mu = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.noise_share_unique_dim, self.mu_logvar_dim),
                self.act
            ) for _ in range(self.num_view)
        )

        # 第七层noise_to_logvar
        self.noise_to_logvar = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.noise_share_unique_dim, self.mu_logvar_dim),
                self.act
            ) for _ in range(self.num_view)
        )

        # 第八层unique_integration
        self.unique_att_1 = nn.TransformerEncoderLayer(d_model=self.unique_integration_dim,
                                                       nhead=self.head_num,
                                                       dim_feedforward=self.unique_integration_dim * 2,
                                                       dropout=self.dropout)

        self.unique_to_cluster = nn.Sequential(
            nn.Linear(self.unique_integration_dim, self.unique_to_cluster_dim),
            self.act
        )

        self.unique_att_2 = nn.TransformerEncoderLayer(d_model=self.unique_to_cluster_dim,
                                                       nhead=self.head_num,
                                                       dim_feedforward=self.unique_to_cluster_dim * 2,
                                                       dropout=self.dropout)

        # 解码器部分
        self.decoder_unique = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.unique_to_cluster_dim, self.noise_share_unique_dim),
                self.act
            ) for _ in range(self.num_view)
        )

        # 第九层
        self.decoder_MLP_1 = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.dec_input_dim, self.hidden_dim[-1]),
                self.act,
                self.dropout_layer
            ) for _ in range(self.num_view)
        )

        # 第十层
        self.decoder_MLP_2 = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.hidden_dim[-1], self.hidden_dim[0]),
                self.act,
                # self.dropout_layer
            ) for _ in range(self.num_view)
        )

        # 第十一层
        self.decoder_MLP_3 = nn.ModuleList(
                nn.Linear(self.hidden_dim[0], self.input_dim[view]) for view in range(self.num_view)
        )

        # 初始化所有的线性层参数
        if self.act_fun != 'gelu':
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.act_fun == 'leakyrelu':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')  # leaky_relu
                elif self.act_fun == 'relu':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif self.act_fun == 'tahn':
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # 模型前向传播
    def forward(self, data):
        list_dict = dict()
        list_dict["noise_list"] = []
        list_dict["share_list"] = []
        list_dict["unique_list"] = []
        list_dict["cluster_data_list"] = []
        list_dict["recon_list"] = []
        list_dict["mu_list"] = []
        list_dict["logvar_list"] = []
        # print(self.cluster_layer)

        # 编码器
        for view in range(self.num_view):
            x = self.encoder_MLP_1[view](data[view])
            x = self.encoder_MLP_2[view](x)

            noise_data = self.noise_star[view](x)
            share_data = self.share_star(x)
            unique_data = self.unique_star[view](x)

            list_dict["noise_list"].append(noise_data)
            list_dict["share_list"].append(share_data)
            list_dict["unique_list"].append(unique_data)

        # 噪音数据重采样
        z_list = []
        for view in range(self.num_view):
            mu_data = self.mu_logvar_bn(self.noise_to_mu[view](list_dict["noise_list"][view]))
            list_dict["mu_list"].append(mu_data)
            logvar_data = self.mu_logvar_bn(self.noise_to_logvar[view](list_dict["noise_list"][view]))
            list_dict["logvar_list"].append(logvar_data)
            z_list.append(self.reparameterization(mu_data, logvar_data))

        # 共享数据整合----隐藏改进点
        stacked_share_list = torch.stack(list_dict["share_list"])
        share_integration = torch.mean(stacked_share_list, dim=0)

        # 独有数据整合
        unique = torch.cat(list_dict["unique_list"], dim=1).unsqueeze(0)
        unique_1 = self.unique_att_1(unique).squeeze(0)
        unique_2 = self.unique_to_cluster(unique_1).unsqueeze(0)
        unique_integration = self.unique_att_2(unique_2).squeeze(0)

        # 聚类数据(潜在表征)
        list_dict["cluster_data_list"].append(torch.cat((share_integration, unique_integration), dim=1))

        # 解码器
        dec_unique_list = []
        for view in range(self.num_view):
            unique_decoder_data = self.decoder_unique[view](unique_integration)
            x = torch.cat((z_list[view], share_integration, unique_decoder_data), dim=1)
            x = self.decoder_MLP_1[view](x)
            x = self.decoder_MLP_2[view](x)
            list_dict["recon_list"].append(self.decoder_MLP_3[view](x))
        return list_dict

    # VAE的重采样
    def reparameterization(self, mu, logvar):
        # 在VAE中，重参数化技巧（reparameterization trick）目的是在训练过程中引入噪声，以便能够计算梯度并通过反向传播进行优化。
        # 而在测试阶段或生成模式下，我们通常直接使用均值 mu 作为潜在变量z，从而生成确定的输出。
        if self.training:
            std = torch.exp(0.5 * logvar)
            # 通过paddle.exp(0.5 * logvar)计算标准差std。这里使用对数方差的一半是因为我们要将其转换为标准差,而对数方差的一半对应于方差的平方根,即标准差。
            eps = torch.randn(std.shape).to(self.device)
            # 从标准正态分布中采样噪声eps,其形状与std相同。这个噪声将用于重参数化。
            return mu + std * eps
            # 通过mu + std * eps计算重参数化后的潜在变量。这个操作将噪声eps乘以标准差std,并加上均值mu,得到从先验分布中采样的潜在变量
        else:
            # 在非训练模式下(self.training为False),即在重构或生成模式下
            # Reconstruction mode
            return mu
            # 直接返回均值mu作为潜在变量的值,不进行重参数化。这是因为在重构或生成模式下,我们通常只关心潜在变量的期望值,而不需要引入随机噪声。

    # T_SNE可视化
    def tsne(self, X):
        model = TSNE(n_components=2)
        return model.fit_transform(X)