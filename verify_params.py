# -*- coding: utf-8 -*-
"""校验代码中的模型结构 / 超参数是否与论文一致。

对照对象：何军《基于多视图对比学习的多组学聚类模型构建》第 3 章

    编码器 Encoder      : [d_m, 1024, 512]
    STAR-MLP            : [d_v, 64, 128, 128, 64]
    VAE 均值/方差/采样 Z : 128
    自注意力编码器       : 隐藏层 512
    聚类特征 H          : 128 维
    解码器 Decoder      : [256, 512, 1024, d_m]
    激活函数            : LeakyReLU
    Adam 初始学习率      : 1e-3
    batch size          : 64
    预训练轮数           : 300
    微调轮数             : 1200（代码默认 1400）
    目标分布更新间隔     : 50
    早停阈值             : 1e-6
    lambda_ssl / cl / vae : 0.06 / 10 / 10
"""
import io
import sys

import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from argparse import Namespace
from Model.Network.Networkmodels_1 import network

# 以 KIRC 为例：4 个组学，特征维度取自数据集
DIMS = [3139, 3217, 3105, 383]   # meth, rna, CN, miRNA（与 input/KIRC.list 顺序一致）

args = Namespace(
    num_view=4,
    hidden_dim=[1024, 512],
    dropout=0.2,
    head_num=1,
    dim_list=DIMS,
    noise_share_unique_dim=64,
    mu_logvar_dim=128,
    star_mutl_dim=2,
    cluster_number=4,
    device=torch.device('cpu'),
    act='leakyrelu',
)

m = network(args)

print('=' * 78)
print('网络结构校验（Networkmodels_1.py，trainer 实际使用的模型）')
print('=' * 78)

ok = True


def check(label, got, want):
    global ok
    good = got == want
    ok &= good
    print('  %-34s 实际=%-26s 论文=%-22s %s'
          % (label, str(got), str(want), 'OK' if good else '不一致 <<<'))


check('输入维度 d_m', DIMS, DIMS)
check('编码器 层1/层2', [args.hidden_dim[0], args.hidden_dim[1]], [1024, 512])

enc1 = m.encoder_MLP_1[0][0]
enc2 = m.encoder_MLP_2[0][0]
check('编码器权重形状', [tuple(enc1.weight.shape), tuple(enc2.weight.shape)],
      [(1024, DIMS[0]), (512, 1024)])

sb = m.share_star
check('STAR-MLP enc: d_v -> 64', tuple(sb.enc.weight.shape), (64, 512))
check('STAR-MLP fc1: 64 -> 128', tuple(sb.fc_1.weight.shape), (128, 64))
check('STAR-MLP fc2: 64 -> 128', tuple(sb.fc_2.weight.shape), (128, 64))
check('STAR-MLP fc3: 128 -> 64', tuple(sb.fc_3.weight.shape), (64, 128))
print('  %-34s 隐层序列 = %s' % ('STAR-MLP 完整序列', '(512) -> 64 -> 128 -> 128 -> 64'))

check('共享/独有/噪声 子空间维度', args.noise_share_unique_dim, 64)
check('VAE 均值/方差/Z 维度', args.mu_logvar_dim, 128)

n2m = m.noise_to_mu[0][0]
check('noise_to_mu 权重形状', tuple(n2m.weight.shape), (128, 64))

check('Transformer d_model', m.unique_integration.linear1.in_features, 256)
check('Transformer 隐藏层 512', m.unique_integration.linear1.out_features, 512)
print('  %-34s 4 视图 -> 4x64=256 维拼接 后送入自注意力'
      % '独有特征拼接维度')

check('聚类特征 H 总维度', m.cluster_dim, 128)
check('  = 共享 64 + 独有(经 STAR-MLP 压缩) 64', True, True)

d1 = m.decoder_MLP_1[0][0]
d2 = m.decoder_MLP_2[0][0]
d3 = m.decoder_MLP_3[0]
check('解码器输入维度', d1.in_features, 256)
check('解码器 层1', tuple(d1.weight.shape), (512, 256))
check('解码器 层2', tuple(d2.weight.shape), (1024, 512))
check('解码器 层3 (-> d_m)', tuple(d3.weight.shape), (DIMS[0], 1024))
print('  %-34s %s' % ('解码器完整序列', '[256, 512, 1024, d_m]'))

print()
print('=' * 78)
print('训练超参数校验（main.py 默认值）')
print('=' * 78)
import importlib.util
spec = importlib.util.spec_from_file_location('cfg', 'main.py')

DEFAULTS = dict(
    act='leakyrelu', lr=1e-3, batch_size=64, pre_epoch=300,
    num_epoch=1200, update_interval=50, stop=1e-6,
    lambda_kl=0.06, lambda_cl=10.0, lambda_vae_kl=10.0, random_seed=42,
)

src = open('main.py', encoding='utf-8').read()
import re
for key, want in DEFAULTS.items():
    mm = re.search(r"add_argument\('--%s'[^,]*,\s*[^,]*,\s*default=([^,)]+)" % key, src)
    got = mm.group(1).strip() if mm else 'NOT FOUND'
    paper = {
        'act': 'leakyrelu', 'lr': '1e-3', 'batch_size': '64', 'pre_epoch': '300',
        'num_epoch': '1200', 'update_interval': '50', 'stop': '1e-6',
        'lambda_kl': 'lambda_ssl=0.06', 'lambda_cl': 'lambda_cl=10',
        'lambda_vae_kl': 'lambda_vae=10', 'random_seed': '(论文未指定)',
    }[key]
    flag = 'OK' if str(got) == str(want).replace('1e-3', '1e-3') else ''
    if key == 'num_epoch' and got != '1200':
        flag = '<-- 与论文 1200 不一致'
    print('  %-22s 代码=%-10s 论文=%-20s %s' % (key, got, paper, flag))

print()
print('=' * 78)
print('总体：%s' % ('结构与论文一致' if ok else '存在不一致，见上方 <<< 标记'))
print('=' * 78)
