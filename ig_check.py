# -*- coding: utf-8 -*-
"""检验积分梯度实现的正确性与稳健性。

1) 完备性公理：Σ IG_i ≈ f(x) − f(baseline)
2) 比较几种「分类输出」目标函数下的归因结果
"""
import argparse
import io
import sys

import numpy as np
import torch

# 注意：integrated_gradients 模块内部已包装 stdout，这里不再重复包装
from integrated_gradients import load_data, build_args
from paths import RESULT_TRAINED_MODEL_DIR, RESULT_CLUSTER_DIR
from Model.Network.Networkmodels_1 import network
from utils import probability_distribution_Q
import pandas as pd


def targets(q, y):
    """返回几种候选的「分类输出」标量。"""
    n = q.shape[0]
    ar = torch.arange(n, device=q.device)
    return {
        'q_assigned': q[ar, y].sum(),
        'margin': (q[ar, y] - q.masked_fill(
            torch.nn.functional.one_hot(y, q.shape[1]).bool(), -1).max(1).values).sum(),
        'logit_assigned': torch.log(q[ar, y] + 1e-12).sum(),
    }


def main(cancer='BRCA', steps=32):
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
    keep = ~lab.isna()
    X = [x[keep.values] for x in X]
    y = torch.tensor(lab.dropna().values.astype(np.int64), device=args.device)
    n = len(y)
    baseline = [torch.zeros_like(x) for x in X]

    # ---- 1) 完备性检查（仅对 q_assigned）
    with torch.no_grad():
        f_x = targets(probability_distribution_Q(
            model(X)['cluster_data_list'][0], model.cluster_layer,
            args.alpha, args.device), y)['q_assigned'].item()
        f_b = targets(probability_distribution_Q(
            model(baseline)['cluster_data_list'][0], model.cluster_layer,
            args.alpha, args.device), y)['q_assigned'].item()

    res = {}
    for name in ('q_assigned', 'margin', 'logit_assigned'):
        avg = {i: torch.zeros_like(X[i]) for i in range(len(X))}
        for a in np.linspace(0, 1, steps):
            inp = [(b + a * (x - b)).detach().requires_grad_(True)
                   for b, x in zip(baseline, X)]
            q = probability_distribution_Q(
                model(inp)['cluster_data_list'][0], model.cluster_layer,
                args.alpha, args.device)
            score = targets(q, y)[name]
            g = torch.autograd.grad(score, inp)
            for i, gg in enumerate(g):
                avg[i] += gg.detach() / steps
        ig_per_mod = {i: ((X[i] - baseline[i]) * avg[i]).detach().cpu().numpy()
                      for i in range(len(X))}
        ig_sum = sum(float(v.sum()) for v in ig_per_mod.values())
        res[name] = ig_per_mod
        print('%-16s Σ IG = %10.3f' % (name, ig_sum))

    tot = sum(float(v.sum()) for v in res['q_assigned'].values())
    print('\n完备性（q_assigned）：f(x) − f(baseline) = %.3f' % (f_x - f_b))
    print('       Σ IG 应接近该值 -> 误差 %.1f%%'
          % (abs(tot - (f_x - f_b)) / max(abs(f_x - f_b), 1e-9) * 100))

    genes = gene_lists['rna']
    tops = {}
    for nm, mods in res.items():
        sc = np.abs(mods[1]).sum(axis=0)
        tops[nm] = set(np.array(genes)[np.argsort(-sc)[:150]])
    print('\n三种目标函数下 top-150 基因的重合度:')
    ks = list(tops)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            print('  %-16s ∩ %-16s = %3d / 150'
                  % (ks[i], ks[j], len(tops[ks[i]] & tops[ks[j]])))
    hejun = ('C1orf64', 'THSD4', 'FOXA1', 'ROPN1', 'TFF3')
    for nm, t in tops.items():
        hit = [g.split('|')[0] for g in t if g.split('|')[0] in hejun]
        print('  %-16s 含何军 Top-5: %s' % (nm, hit or '无'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cancer', default='BRCA')
    ap.add_argument('--steps', type=int, default=32)
    a = ap.parse_args()
    main(a.cancer, a.steps)
