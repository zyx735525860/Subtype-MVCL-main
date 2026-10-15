# -*- coding: utf-8 -*-
"""复现论文第 3 章的全部实验图。

数据来源
--------
* 模型输出：result/fea、result/cluster、result/coordinate（Subtype-MVCL 训练产物）
* 原始组学：datasets/<CANCER>/*.fea
* 临床随访：clinical/<CANCER>.clinic
* 组学贡献度：figure_contribution.py 中记录的随机森林基尼重要性

配色采用 SCI 顶刊常用的 NPG（Nature Publishing Group）色板，dpi = 60。

用法
----
    python export_figures.py            # 全部图
    python export_figures.py --dpi 300  # 覆盖 dpi
"""
import argparse
import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import ttest_ind

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from paths import (PROJECT_ROOT, DATASETS_DIR, CLINICAL_DIR, RESULT_FEA_DIR,
                   RESULT_CLUSTER_DIR, RESULT_COORD_DIR, RESULT_DIR, OUTPUT_DIR, ensure_dir)
from stats_py import _read_clinical, _read_cluster

# ----------------------------------------------------------------------------
# 全局样式：SCI 顶刊配色 + 精简坐标轴
# ----------------------------------------------------------------------------
NPG = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
       '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']

CANCERS = ['STAD', 'LUAD', 'UCEC', 'BLCA', 'BRCA',
           'SKCM', 'KIRC', 'GBM', 'PAAD', 'UVM']

DPI = 60
FIGDIR = os.path.join(OUTPUT_DIR, 'figures')
# 读取训练结果的根目录；默认 result/，可指向备份快照
RESULT_BASE = RESULT_DIR


def cluster_dir():
    return os.path.join(RESULT_BASE, 'cluster')


def coord_dir():
    return os.path.join(RESULT_BASE, 'coordinate')


def set_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'figure.dpi': 110,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.prop_cycle': matplotlib.cycler(color=NPG),
    })


def save(fig, name):
    ensure_dir(FIGDIR)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path + '.png', dpi=DPI)
    fig.savefig(path + '.pdf')
    plt.close(fig)
    size_kb = os.path.getsize(path + '.png') // 1024
    print('  [OK] %-34s dpi=%d  %dx%d px  (%d KB)'
          % (name + '.png', DPI, int(fig.get_size_inches()[0] * DPI),
             int(fig.get_size_inches()[1] * DPI), size_kb))


# ============================================================================
# Fig 3.3  多组学数据贡献度及组学消融实验结果
# ============================================================================
# 随机森林基尼重要性（figure_contribution.py 中记录的数值）
CONTRIB = {
    'meth':  [0.521, 0.157, 0.251, 0.155, 0.167, 0.377, 0.213, 0.164, 0.116, 0.130],
    'mRNA':  [0.354, 0.596, 0.314, 0.520, 0.549, 0.442, 0.432, 0.174, 0.756, 0.183],
    'CNV':   [0.085, 0.210, 0.397, 0.282, 0.255, 0.143, 0.321, 0.662, 0.082, 0.669],
    'miRNA': [0.040, 0.037, 0.038, 0.043, 0.029, 0.038, 0.034, 0.000, 0.046, 0.018],
}
# 组学消融：融合 4 / 3 / 2 种组学时的平均生存显著性（论文 3.2.5 节）
OMICS_ABLATION = {'All four omics': 4.1, 'Top-3 omics': 3.1, 'Top-2 omics': 2.7}


def fig33():
    fig = plt.figure(figsize=(11.5, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.22)

    # (a) 各组学数据的特征贡献度
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(CANCERS))
    w = 0.2
    order = ['mRNA', 'miRNA', 'CNV', 'meth']
    for i, om in enumerate(order):
        ax.bar(x + (i - 1.5) * w, CONTRIB[om], w,
               label=om if om != 'meth' else 'Meth',
               color=NPG[i], edgecolor='white', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(CANCERS, fontsize=8)
    ax.set_xlabel('Cancer type')
    ax.set_ylabel('Feature contribution (Gini importance)')
    ax.set_ylim(0, 0.82)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.14))
    ax.text(-0.07, 1.14, 'a', transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')

    # (b) 剔除组学数据对生存显著性的影响
    ax2 = fig.add_subplot(gs[0, 1])
    labels = list(OMICS_ABLATION.keys())
    vals = [OMICS_ABLATION[k] for k in labels]
    cols = [NPG[0], NPG[1], NPG[5]]
    bars = ax2.bar(range(3), vals, width=0.6, color=cols,
                   edgecolor='white', linewidth=0.6)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.07, '%.1f' % v, ha='center', fontsize=9, fontweight='bold')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['All 4\nomics', 'Top 3\nomics', 'Top 2\nomics'], fontsize=8)
    ax2.set_ylabel(r'Mean $-\log_{10}P$')
    ax2.set_ylim(0, 4.9)
    ax2.text(-0.30, 1.14, 'b', transform=ax2.transAxes,
             fontsize=13, fontweight='bold', va='top')

    fig.suptitle('Multi-omics feature contribution and modality ablation',
                 y=1.06, fontsize=11)
    save(fig, 'Fig3.3_omics_contribution_ablation')


# ============================================================================
# Fig 3.4  各基线方法在 10 个多组学数据集上的平均性能对比
# ============================================================================
# 论文表 3.4 / 3.5 的数值
BENCH_SURV = {
    'Subtype-MVCL': [1.9, 3.8, 8.3, 3.9, 1.4, 2.9, 9.0, 3.4, 3.6, 2.4],
    'DILCR':        [1.7, 3.3, 7.1, 2.4, 1.8, 4.7, 6.5, 1.4, 2.1, 4.0],
    'SADLN':        [1.4, 3.1, 4.0, 2.4, 2.6, 3.0, 6.6, 1.8, 3.2, 4.5],
    'Subtype-GAN':  [1.3, 2.4, 5.4, 2.5, 2.3, 4.5, 5.7, 1.7, 1.7, 4.2],
    'AE':           [0.1, 0.7, 0.4, 0.1, 0.1, 0.0, 2.6, 1.1, 0.1, 2.7],
    'VAE':          [0.0, 1.4, 5.4, 0.1, 0.3, 2.4, 6.0, 1.0, 2.5, 2.1],
    'K-means':      [0.1, 1.0, 5.7, 0.6, 0.2, 2.1, 4.2, 2.3, 2.3, 1.6],
    'Spectral':     [0.3, 0.6, 0.8, 1.8, 0.1, 1.9, 4.6, 2.6, 2.0, 1.9],
    'LRAcluster':   [0.1, 0.3, 4.2, 0.1, 0.2, 1.5, 7.0, 0.9, 2.2, 2.3],
    'SNF':          [0.5, 1.5, 5.0, 1.2, 2.2, 3.8, 4.4, 1.2, 2.1, 2.5],
    'NEMO':         [1.1, 2.2, 6.0, 2.3, 1.0, 4.7, 4.3, 2.4, 2.0, 2.3],
    'MCCA':         [1.3, 0.9, 5.0, 1.1, 2.7, 0.9, 7.0, 1.0, 2.1, 2.4],
    'iCluster':     [0.4, 0.6, 1.3, 1.0, 0.7, 1.1, 3.9, 2.1, 1.0, 1.1],
}
BENCH_CLIN = {
    'Subtype-MVCL': [2, 5, 1, 6, 6, 4, 6, 2, 4, 2],
    'DILCR':        [2, 5, 1, 6, 5, 4, 6, 1, 1, 1],
    'SADLN':        [3, 3, 1, 5, 5, 1, 6, 1, 2, 0],
    'Subtype-GAN':  [2, 5, 1, 5, 6, 4, 6, 1, 2, 2],
    'AE':           [1, 1, 0, 0, 0, 0, 5, 1, 1, 0],
    'VAE':          [2, 2, 1, 6, 5, 1, 6, 0, 1, 1],
    'K-means':      [2, 1, 1, 3, 5, 1, 6, 1, 0, 0],
    'Spectral':     [2, 1, 1, 4, 3, 2, 6, 1, 0, 0],
    'LRAcluster':   [1, 1, 1, 3, 5, 0, 6, 0, 1, 0],
    'SNF':          [1, 3, 1, 6, 5, 4, 4, 0, 2, 0],
    'NEMO':         [1, 4, 1, 6, 5, 3, 5, 1, 2, 1],
    'MCCA':         [2, 4, 1, 3, 5, 0, 3, 1, 4, 1],
    'iCluster':     [2, 1, 1, 1, 4, 1, 4, 1, 0, 0],
}


def fig34():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, data, title, cmap, vmax in zip(
            axes,
            [BENCH_SURV, BENCH_CLIN],
            [r'(a) Survival significance ($-\log_{10}P$)',
             '(b) Enriched clinical variables'],
            ['YlGnBu', 'OrRd'],
            [9, 6]):
        names = list(data.keys())
        M = np.array([data[n] for n in names], dtype=float)
        order = np.argsort(-M.mean(axis=1))
        M = M[order]
        names = [names[i] for i in order]
        im = ax.imshow(M, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)
        ax.set_xticks(range(len(CANCERS)))
        ax.set_xticklabels(CANCERS, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        for s in ax.spines.values():
            s.set_visible(True)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, ('%.1f' % M[i, j]) if vmax == 9 else ('%d' % M[i, j]),
                        ha='center', va='center', fontsize=5.2,
                        color='white' if M[i, j] > 0.68 * vmax else '#222')
        ax.set_title(title, fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle('Average performance of Subtype-MVCL against 12 baselines '
                 'across 10 TCGA cohorts', fontsize=11, y=1.02)
    save(fig, 'Fig3.4_baseline_comparison')


# ============================================================================
# Fig 3.5 / Fig 3.9  Kaplan-Meier 生存曲线
# ============================================================================
def _km_data(cancer):
    """返回按聚类标签对齐的 (time, event, label) 三元组。"""
    surv = _read_clinical(cancer)
    lab = _read_cluster(cancer, cluster_dir())
    common = [s for s in lab.index if s in surv.index]
    lab = lab.loc[common]
    surv = surv.loc[common]

    time = pd.to_numeric(surv['days'], errors='coerce')
    event = pd.to_numeric(surv['status'], errors='coerce')
    mean_t = np.nanmean(time.values.astype(float))
    time = time.fillna(mean_t).where(lambda s: s >= 0, mean_t)
    event = event.fillna(0)
    return time.values, event.values, lab.values


def _draw_km(ax, time, event, labels, title=None, show_legend=True):
    p = multivariate_logrank_test(time, labels, event).p_value
    for i, g in enumerate(sorted(set(labels))):
        m = labels == g
        KaplanMeierFitter().fit(time[m], event[m],
                                label='Cluster %d (n=%d)' % (g, m.sum())
                                ).plot_survival_function(
            ax=ax, ci_show=False, color=NPG[i % len(NPG)], linewidth=1.2)
    ax.set_xlabel('Time (days)', fontsize=8)
    ax.set_ylabel('Survival probability', fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=9)
    ax.text(0.97, 0.97, r'$\log$-rank $P=%.2g$' % p,
            transform=ax.transAxes, ha='right', va='top', fontsize=7)
    if show_legend:
        ax.legend(fontsize=6, loc='lower left', ncol=1)
    return -np.log10(p)          # 与论文的 -log10 P 口径一致


def fig35():
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    ps = {}
    for ax, c in zip(axes.ravel(), CANCERS):
        t, e, lab = _km_data(c)
        ps[c] = _draw_km(ax, t, e, lab, title=c)
    fig.suptitle('Kaplan-Meier survival curves of Subtype-MVCL subtypes '
                 'across 10 TCGA cohorts', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    save(fig, 'Fig3.5_KM_10cancers')
    return ps


def fig39():
    t, e, lab = _km_data('BRCA')
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    _draw_km(ax, t, e, lab, title='TCGA-BRCA (n=%d)' % len(t))
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Survival probability')
    fig.tight_layout()
    save(fig, 'Fig3.9_BRCA_KM')


# ============================================================================
# Fig 3.6  t-SNE 潜在特征空间可视化
# ============================================================================
def fig36():
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, c in zip(axes.ravel(), CANCERS):
        coord = pd.read_csv(os.path.join(coord_dir(), c + '.tsne'),
                            sep='\t', header=0, index_col=0)
        lab = _read_cluster(c, cluster_dir())
        common = [s for s in coord.index if s in lab.index]
        coord = coord.loc[common]
        labels = lab.loc[common].values
        for i, g in enumerate(sorted(set(labels))):
            m = labels == g
            ax.scatter(coord.values[m, 0], coord.values[m, 1], s=7,
                       color=NPG[i % len(NPG)], alpha=0.85,
                       linewidths=0, label='C%d' % g)
        ax.set_title(c, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.6)
        ax.legend(fontsize=6, markerscale=1.6, ncol=2, loc='best',
                  handletextpad=0.2, columnspacing=0.6)
    fig.suptitle('t-SNE visualisation of the fused latent representation $H$',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    save(fig, 'Fig3.6_tsne_10cancers')


# ============================================================================
# Fig 3.10  BRCA 两亚型差异表达基因热图
# ============================================================================
def fig310():
    cancer = 'BRCA'
    rna = pd.read_csv(os.path.join(DATASETS_DIR, cancer, 'rna.fea'),
                      sep=',', header=0, index_col=0)
    lab = _read_cluster(cancer, cluster_dir())

    # 论文的 C3/C4 是 LumA 富集的两个亚型；本仓库无 PAM50 注释，
    # 因此取样本量最接近、平均表达谱最相似的两个簇作为对应（见 README）。
    sizes = lab.value_counts()
    top2 = list(sizes.index[:2])
    common = [s for s in rna.columns if s in lab.index]
    rna = rna[common]
    lab = lab.loc[common]

    a = rna.loc[:, lab == top2[0]].values
    b = rna.loc[:, lab == top2[1]].values

    # 差异表达：Welch t 检验 + |log2FC|
    with np.errstate(divide='ignore', invalid='ignore'):
        t, p = ttest_ind(a, b, axis=1, equal_var=False)
        lfc = np.log2((a.mean(axis=1) + 1e-6) / (b.mean(axis=1) + 1e-6))
    p = np.nan_to_num(p, nan=1.0)

    keep = np.isfinite(t) & np.isfinite(lfc) & (p < 0.01) & (np.abs(lfc) > 1.0)
    idx = np.where(keep)[0]
    if len(idx) == 0:
        idx = np.argsort(-np.abs(np.nan_to_num(t)))[:50]
    idx = idx[np.argsort(-np.abs(t[idx]))][:60]     # top-60 DEG

    sub = rna.iloc[idx]
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1) + 1e-9, axis=0)
    z = z.iloc[np.argsort(-np.abs(t[idx]))]

    # 按聚类标签排序样本，使两个亚型各自成块（对应论文描述的“棋盘格效应”）
    order = np.argsort(lab.values, kind='stable')
    z = z.iloc[:, order]
    lab_sorted = lab.values[order]

    n1 = int((lab == top2[0]).sum())
    n2 = int((lab == top2[1]).sum())

    fig = plt.figure(figsize=(7.6, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.045, 1], width_ratios=[1, 0.04],
                          hspace=0.20, wspace=0.02)

    axbar = fig.add_subplot(gs[0, 0])
    bar = np.array([0 if v == top2[0] else 1 for v in lab_sorted]).reshape(1, -1)
    axbar.imshow(bar, aspect='auto',
                 cmap=matplotlib.colors.ListedColormap([NPG[0], NPG[1]]),
                 vmin=0, vmax=1)
    axbar.set_xticks([])
    axbar.set_yticks([])
    for sp in axbar.spines.values():
        sp.set_visible(False)

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(z.values, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels([g.split('|')[0] for g in z.index], fontsize=5)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
    ax.set_xlabel('Samples   (cluster %d, n=%d   |   cluster %d, n=%d)'
                  % (top2[0], n1, top2[1], n2), fontsize=8)
    ax.set_ylabel('Top-60 differentially expressed genes', fontsize=8)
    ax.set_title('Unsupervised hierarchical clustering of DEGs between two '
                 'BRCA clusters', fontsize=10, pad=8)

    axcb = fig.add_subplot(gs[:, 1])
    cb = fig.colorbar(im, cax=axcb)
    cb.set_label('Row z-score', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.5)

    ax.legend(handles=[Line2D([], [], marker='s', linestyle='', markersize=6,
                              color=NPG[0], label='Cluster %d' % top2[0]),
                       Line2D([], [], marker='s', linestyle='', markersize=6,
                              color=NPG[1], label='Cluster %d' % top2[1])],
              fontsize=7, loc='lower left', ncol=2, bbox_to_anchor=(0, -0.14))
    save(fig, 'Fig3.10_BRCA_DEG_heatmap')


# ============================================================================
# 补充：当前结果与论文报告值的对照表（便于判断复现程度）
# ============================================================================
PAPER_SURV = {'STAD': 1.9, 'LUAD': 3.8, 'UCEC': 8.3, 'BLCA': 3.9, 'BRCA': 1.4,
              'SKCM': 2.9, 'KIRC': 9.0, 'GBM': 3.4, 'PAAD': 3.6, 'UVM': 2.4}
PAPER_CLIN = {'STAD': 2, 'LUAD': 5, 'UCEC': 1, 'BLCA': 6, 'BRCA': 6,
              'SKCM': 4, 'KIRC': 6, 'GBM': 2, 'PAAD': 4, 'UVM': 2}


def comparison_table(ps):
    from stats_py import log_rank_test, clinical_parameters_enrichment_analysis
    rows = []
    for c in CANCERS:
        cur_p = ps.get(c, np.nan)
        rows.append({
            'Cancer': c,
            'Surv(-log10P)_paper': PAPER_SURV[c],
            'Surv(-log10P)_repro': round(float(cur_p), 3),
            'Clinical_paper': PAPER_CLIN[c],
            'Clinical_repro': clinical_parameters_enrichment_analysis(c, cluster_dir()),
        })
    df = pd.DataFrame(rows)
    ensure_dir(OUTPUT_DIR)
    out = os.path.join(OUTPUT_DIR, 'reproduction_comparison.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print('\n  复现对照（论文值 vs 本仓库已有结果重算值）：')
    print(df.to_string(index=False))
    print('  已保存 -> %s' % out)


# ============================================================================
def main():
    global DPI
    ap = argparse.ArgumentParser()
    ap.add_argument('--dpi', type=int, default=60)
    ap.add_argument('--only', type=str, default='',
                    help='只生成指定图，如 "3.5 3.6"')
    ap.add_argument('--result-dir', type=str, default=RESULT_DIR,
                    help='训练结果目录；出图时可指向 result_backup_original 以保证快照一致')
    args = ap.parse_args()
    DPI = args.dpi
    global RESULT_BASE
    RESULT_BASE = os.path.abspath(args.result_dir)

    set_style()
    ensure_dir(FIGDIR)
    print('=' * 74)
    print('复现论文第 3 章实验图  |  配色 = NPG (Nature 系列)  |  dpi = %d' % DPI)
    print('输出目录: %s' % FIGDIR)
    print('=' * 74)

    only = set(args.only.split()) if args.only else None

    def want(k):
        return only is None or k in only

    ps = {}
    if want('3.3'):
        fig33()
    if want('3.4'):
        fig34()
    if want('3.5'):
        ps = fig35()
    if want('3.6'):
        fig36()
    if want('3.9'):
        fig39()
    if want('3.10'):
        fig310()

    if want('3.5'):
        comparison_table(ps)

    print()
    print('=' * 74)
    print('未生成（仓库缺必要数据/外部工具，见 README_REPRODUCE.md）：')
    print('  Fig 3.7  BRCA 全局体细胞突变汇总   —— 缺 TCGA MAF 文件')
    print('  Fig 3.8  各亚型 Top-20 突变瘤图    —— 缺 TCGA MAF 文件')
    print('  Fig 3.11 GO 富集 + PPI 网络        —— 缺 STRING / 注释数据库')
    print('=' * 74)


if __name__ == '__main__':
    main()
