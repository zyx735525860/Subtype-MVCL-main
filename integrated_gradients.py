# -*- coding: utf-8 -*-
"""对训练好的 Subtype-MVCL 模型做积分梯度（Integrated Gradients）归因。

原仓库只有基于随机森林的 biomarkers.py，没有梯度归因实现。本脚本按论文
§3.2.8 的描述补齐：沿「全零基线 → 真实样本」的线性插值路径累加梯度，
量化每个 mRNA 特征对聚类分配输出的贡献。

用法：
    python integrated_gradients.py BRA     # 指定癌种
    python integrated_gradients.py BRCA --steps 64 --topk 150
"""
import argparse
import io
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from paths import DATASETS_DIR, INPUT_DIR, RESULT_CLUSTER_DIR, RESULT_TRAINED_MODEL_DIR
from Model.Network.Networkmodels_1 import network
from utils import probability_distribution_Q

CANCER_K = {'BRCA': 5, 'BLCA': 5, 'KIRC': 4, 'GBM': 4, 'LUAD': 3,
            'PAAD': 2, 'SKCM': 4, 'STAD': 3, 'UCEC': 4, 'UVM': 4}


def build_args(cancer, dims):
    return argparse.Namespace(
        num_view=4, hidden_dim=[1024, 512], dropout=0.2, head_num=1,
        dim_list=dims, noise_share_unique_dim=64, mu_logvar_dim=128,
        star_mutl_dim=2, cluster_number=CANCER_K[cancer],
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        act='leakyrelu', alpha=1.0,
    )


def load_data(cancer):
    """按 input/<CANCER>.list 的顺序读取并标准化四种组学（不加噪声）。"""
    order, mats, gene_lists = [], [], {}
    for line in open(os.path.join(INPUT_DIR, cancer + '.list'), 'rt'):
        base = os.path.splitext(os.path.basename(line.strip()))[0]
        order.append(base)
        df = pd.read_csv(os.path.join(DATASETS_DIR, cancer, base + '.fea'),
                         sep=',', header=0, index_col=0)
        gene_lists[base] = list(df.index)          # 每个组学各自的特征名
        df = df.T
        df = (df - df.mean()) / df.std()
        mats.append(df.values.astype(np.float32))
    samples = list(df.index)
    return order, mats, samples, gene_lists


def run_ig(cancer, steps=64, topk=150):
    order, mats, samples, gene_lists = load_data(cancer)
    dims = [m.shape[1] for m in mats]
    print('[%s] 组学顺序 %s，维度 %s，样本 %d' % (cancer, order, dims, len(samples)))

    args = build_args(cancer, dims)
    model = network(args)
    state = torch.load(os.path.join(RESULT_TRAINED_MODEL_DIR, cancer + '.pth'),
                       map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.to(args.device).eval()

    X = [torch.tensor(m, device=args.device) for m in mats]
    labels = pd.read_csv(os.path.join(RESULT_CLUSTER_DIR, cancer + '.cluster'),
                         sep='\t', header=0, index_col=0)['Labels']
    labels = labels.reindex(samples)
    if labels.isna().any():
        keep = [i for i, s in enumerate(samples) if s in labels.index]
        X = [x[keep] for x in X]
        samples = [samples[i] for i in keep]
        labels = labels.dropna()
    y = torch.tensor(labels.values.astype(np.int64), device=args.device)
    n = len(samples)

    # 全零基线；沿插值路径累加梯度
    baseline = [torch.zeros_like(x) for x in X]
    avg_grad = {i: torch.zeros_like(X[i]) for i in range(len(X))}
    for a in np.linspace(0, 1, steps):
        inp = [(b + a * (x - b)).detach().requires_grad_(True)
               for b, x in zip(baseline, X)]
        out = model(inp)
        H = out['cluster_data_list'][0]
        q = probability_distribution_Q(H, model.cluster_layer, args.alpha, args.device)
        score = q[torch.arange(n, device=args.device), y].sum()
        grads = torch.autograd.grad(score, inp, retain_graph=False)
        for i, g in enumerate(grads):
            avg_grad[i] += g.detach() / steps
        if (int(round(a * (steps - 1))) + 1) % max(1, steps // 4) == 0:
            print('  插值步 %d/%d' % (int(round(a * (steps - 1))) + 1, steps))

    result = {}
    for i, name in enumerate(order):
        ig = (X[i] - baseline[i]) * avg_grad[i]
        result[name] = ig.detach().cpu().numpy()
    return result, samples, gene_lists, labels.values


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cancer', nargs='?', default='BRCA')
    ap.add_argument('--steps', type=int, default=64)
    ap.add_argument('--topk', type=int, default=150)
    ap.add_argument('--outdir', default='outputs/ig')
    a = ap.parse_args()

    res, samples, gene_lists, labels = run_ig(a.cancer, a.steps)
    os.makedirs(a.outdir, exist_ok=True)

    # mRNA 归因（组学顺序中 index=1）
    mrna = res['rna']
    genes = gene_lists['rna']
    score = np.abs(mrna).sum(axis=0)           # 每个基因在所有样本上的绝对贡献
    rank = pd.DataFrame({'gene': genes, 'IG_score': score}) \
             .sort_values('IG_score', ascending=False).reset_index(drop=True)
    rank.to_csv(os.path.join(a.outdir, '%s_IG_all_genes.csv' % a.cancer), index=False)
    top = rank.head(a.topk)
    top.to_csv(os.path.join(a.outdir, '%s_IG_top%d.csv' % (a.cancer, a.topk)), index=False)

    print('\nTop-15 基因（含 IG 得分）：')
    for i, r in top.head(15).iterrows():
        print('  %2d. %-14s %.4f' % (i + 1, str(r['gene']).split('|')[0], r['IG_score']))

    # 每个亚型的 top 基因，供后续分型分析
    sub_rows = []
    for c in sorted(set(labels)):
        m = labels == c
        s = np.abs(mrna[m]).sum(axis=0)
        idx = np.argsort(-s)[:20]
        for i in idx:
            sub_rows.append({'cluster': int(c), 'gene': genes[i], 'IG_score': float(s[i])})
    pd.DataFrame(sub_rows).to_csv(
        os.path.join(a.outdir, '%s_IG_per_cluster_top20.csv' % a.cancer), index=False)

    print('\n已写入 %s/' % a.outdir)
