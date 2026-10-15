# -*- coding: utf-8 -*-
"""用 margin 目标（q[y] − max_{j≠y} q[j]）重算 BRCA 归因，并导出 top-150，
以检验它是否比 q[y] 更接近论文报告的生物学（ECM / 细胞黏附）。"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from integrated_gradients import load_data, build_args
from paths import RESULT_TRAINED_MODEL_DIR, RESULT_CLUSTER_DIR
from Model.Network.Networkmodels_1 import network
from utils import probability_distribution_Q

OUTDIR = 'outputs/ig'
os.makedirs(OUTDIR, exist_ok=True)


def run(cancer='BRCA', steps=32, topk=150):
    order, mats, samples, gene_lists = load_data(cancer)
    args = build_args(cancer, [m.shape[1] for m in mats])
    model = network(args)
    model.load_state_dict(torch.load(
        RESULT_TRAINED_MODEL_DIR + '/%s.pth' % cancer,
        map_location='cpu', weights_only=True))
    model.to(args.device).eval()

    X = [torch.tensor(m, device=args.device) for m in mats]
    lab = pd.read_csv(RESULT_CLUSTER_DIR + '/%s.cluster' % cancer,
                      sep='\t', header=0, index_col=0)['Labels'].reindex(samples)
    keep = (~lab.isna()).values
    X = [x[keep] for x in X]
    y = torch.tensor(lab.dropna().values.astype(np.int64), device=args.device)
    n = len(y)
    baseline = [torch.zeros_like(x) for x in X]
    onehot = F.one_hot(y, args.cluster_number).bool()

    avg = {i: torch.zeros_like(X[i]) for i in range(len(X))}
    for a in np.linspace(0, 1, steps):
        inp = [(b + a * (x - b)).detach().requires_grad_(True)
               for b, x in zip(baseline, X)]
        q = probability_distribution_Q(
            model(inp)['cluster_data_list'][0], model.cluster_layer,
            args.alpha, args.device)
        other = q.masked_fill(onehot, -1).max(1).values
        score = (q[torch.arange(n, device=args.device), y] - other).sum()
        g = torch.autograd.grad(score, inp)
        for i, gg in enumerate(g):
            avg[i] += gg.detach() / steps

    ig = (X[1] - baseline[1]) * avg[1]
    sc = np.abs(ig.detach().cpu().numpy()).sum(axis=0)
    genes = gene_lists['rna']
    rank = pd.DataFrame({'gene': genes, 'IG_score': sc}).sort_values(
        'IG_score', ascending=False).reset_index(drop=True)
    rank.to_csv(os.path.join(OUTDIR, '%s_IG_margin_all.csv' % cancer), index=False)
    rank.head(topk).to_csv(
        os.path.join(OUTDIR, '%s_IG_margin_top%d.csv' % (cancer, topk)), index=False)
    sym = [g.split('|')[0] for g in rank.head(10)['gene']]
    print('margin 目标 top-10:', sym)
    hj = ('C1orf64', 'THSD4', 'FOXA1', 'ROPN1', 'TFF3')
    r = {g.split('|')[0]: i + 1 for i, g in enumerate(rank['gene'])}
    print('何军 Top-5 排名:', {g: r.get(g, '—') for g in hj})


if __name__ == '__main__':
    run()
