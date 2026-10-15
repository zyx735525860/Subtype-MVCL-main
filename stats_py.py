# -*- coding: utf-8 -*-
"""纯 Python 替代 rpy2 + R 的统计计算。

原代码在 Models_trainer.py 中通过 `rpy2.robjects.r(...)` 调用 R 的
survival / stats 包完成两件事：

1. `log_rank_test(subtype)`        —— 单变量 log-rank 检验，返回 -log10(p)
2. `clinical_parameters_enrichment_analysis(subtype)` —— 临床参数富集，返回
   p < 0.05 的参数个数

本模块用 lifelines + scipy 复现同样的语义，不再依赖 R 与 rpy2 的安装。
两处行为的对应关系：

    R survdiff(Surv(time, event) ~ cluster)  -> lifelines multivariate_logrank_test
    R chisq.test(table(cluster, var))        -> scipy.stats.chi2_contingency
                                                 (2x2 表同样启用 Yates 连续性校正)
    R kruskal.test(value, cluster)           -> scipy.stats.kruskal

临床变量的清洗规则（pathologic_M/N/T/stage 的字符串归一化、缺失比例超过 50%
则跳过、离散变量只有单一水平则跳过）与原 R 代码保持一致。
"""
import os

import numpy as np
import pandas as pd
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import chi2_contingency, kruskal

from paths import CLINICAL_DIR, RESULT_CLUSTER_DIR

# 需要检验的临床变量及其类型（对应原 R 代码中的 clinical_metadata）
CLINICAL_METADATA = [
    ('gender', 'DISCRETE'),
    ('age_at_initial_pathologic_diagnosis', 'NUMERIC'),
    ('pathologic_M', 'DISCRETE'),
    ('pathologic_N', 'DISCRETE'),
    ('pathologic_T', 'DISCRETE'),
    ('pathologic_stage', 'DISCRETE'),
]

# 允许的取值集合（对应原 R 代码里对各分期字段的归一化白名单）
ALLOWED = {
    'pathologic_M': {'MX', 'M0', 'M1'},
    'pathologic_T': {'TX', 'T0', 'T1', 'T2', 'T3', 'T4', 'Tis'},
    'pathologic_N': {'NX', 'N0', 'N1', 'N2', 'N3'},
    'pathologic_stage': {'Stage X', 'Stage 0', 'Stage I', 'Stage II', 'Stage III', 'Stage IV'},
}

# 各分期字段需要剥掉的字母后缀（对应 R 里的 gsub 链）
STRIP = {
    'pathologic_M': ['a', 'b', 'c'],
    'pathologic_T': ['a', 'b', 'c', 'd'],
    'pathologic_N': ['a', 'b', 'c', 'mi'],
}


def _read_clinical(subtype):
    """读取临床表，行为与 R 的 `read.table(..., row.names=1)` 一致。"""
    path = os.path.join(CLINICAL_DIR, subtype + '.clinic')
    df = pd.read_csv(path, header=0)
    df = df.set_index(df.columns[0])
    return df


def _read_cluster(subtype, cluster_dir=None):
    """读取聚类标签，以样本条码为索引。

    cluster_dir 默认取 result/cluster；出图时可指向备份快照
    （result_backup_original/cluster）以保证与之前生成的图内容一致。
    """
    path = os.path.join(cluster_dir or RESULT_CLUSTER_DIR, subtype + '.cluster')
    df = pd.read_csv(path, sep='\t', header=0, index_col=0)
    return df['Labels']


def log_rank_test(subtype, cluster_dir=None):
    """返回 -log10(p)，与原 R 实现一致。

    R 代码在检验前会把缺失/为负的生存时间替换为均值、缺失的死亡状态置 0，
    这里保持同样的预处理顺序。
    """
    survival = _read_clinical(subtype)
    labels = _read_cluster(subtype, cluster_dir)

    # 按聚类标签的样本顺序对齐临床表（对应 R 的 match()）
    missing = [s for s in labels.index if s not in survival.index]
    if missing:
        raise ValueError('%s: %d 个聚类样本在临床表中缺失' % (subtype, len(missing)))
    ordered = survival.loc[labels.index].copy()

    death = pd.to_numeric(ordered['status'], errors='coerce')
    time = pd.to_numeric(ordered['days'], errors='coerce')

    # R: Survival[is.na | <0] <- mean(Survival); Death[is.na] <- 0
    mean_value = np.nanmean(time.values.astype(float))
    time = time.fillna(mean_value)
    time = time.where(time >= 0, mean_value)
    death = death.fillna(0)

    res = multivariate_logrank_test(time.values, labels.values, death.values)
    return -np.log10(res.p_value)


def _clean_discrete(series, param):
    """复现原 R 代码里对分期字段的字符串归一化。"""
    s = series.astype(str).str.strip()

    if param == 'pathologic_M':
        for ch in STRIP['pathologic_M']:
            s = s.str.replace(ch, '', regex=False)
        s = s.str.replace(r' \(i\+\)', '', regex=True)
    elif param == 'pathologic_T':
        s = s.replace('[Discrepancy]', np.nan)
        for ch in STRIP['pathologic_T']:
            s = s.str.replace(ch, '', regex=False)
    elif param == 'pathologic_N':
        for ch in STRIP['pathologic_N']:
            s = s.str.replace(ch, '', regex=False)
        s = s.str.replace(r' \(i\+\)', '', regex=True)
        s = s.str.replace(r' \(i-\)', '', regex=True)
        s = s.str.replace(r' \(mol\+\)', '', regex=True)
    elif param == 'pathologic_stage':
        s = s.replace('[Discrepancy]', np.nan)
        for ch in ('A', 'B', 'C'):
            s = s.str.replace(ch, '', regex=False)

    allowed = ALLOWED.get(param)
    if allowed is not None:
        s = s.where(s.isin(allowed), np.nan)
    return s


def _pvalue_discrete(cluster, values):
    """离散变量：卡方检验（2x2 表自动启用 Yates 连续性校正，与 R 一致）。"""
    frame = pd.DataFrame({'cluster': cluster, 'value': values}).dropna()
    table = pd.crosstab(frame['cluster'], frame['value'])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    try:
        _, p, _, _ = chi2_contingency(table.values)
    except ValueError:
        return None
    return p


def _pvalue_numeric(cluster, values):
    """连续变量：Kruskal-Wallis 检验。"""
    frame = pd.DataFrame({'cluster': cluster, 'value': values}).dropna()
    groups = [g['value'].values for _, g in frame.groupby('cluster')]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return None
    try:
        _, p = kruskal(*groups)
    except ValueError:
        return None
    return p


def clinical_parameters_enrichment_analysis(subtype, cluster_dir=None):
    """返回显著（p < 0.05）的临床参数个数，与原 R 实现一致。"""
    clinical = _read_clinical(subtype)
    cluster = _read_cluster(subtype, cluster_dir)

    # 只保留两边都有的样本，并按聚类结果对齐
    common = [s for s in cluster.index if s in clinical.index]
    cluster = cluster.loc[common]

    pvalues = []
    for param, kind in CLINICAL_METADATA:
        if param not in clinical.columns:
            continue

        values = clinical.loc[common, param]

        if kind == 'DISCRETE':
            values = _clean_discrete(values, param)

        # R: 缺失比例超过一半则跳过
        not_na = values.notna()
        if 2 * not_na.sum() < len(values):
            continue

        if kind == 'DISCRETE':
            # R: 离散变量只剩单一水平则跳过
            if values[not_na].nunique() < 2:
                continue
            p = _pvalue_discrete(cluster, values)
        else:
            values = pd.to_numeric(values, errors='coerce')
            not_na = values.notna()
            if 2 * not_na.sum() < len(values):
                continue
            p = _pvalue_numeric(cluster, values)

        if p is not None and not np.isnan(p):
            pvalues.append(p)

    return int(sum(1 for p in pvalues if p < 0.05))


if __name__ == '__main__':
    # 自检：对已有结果逐癌种计算，与论文表格对照
    for c in ['STAD', 'LUAD', 'UCEC', 'BLCA', 'BRCA',
              'SKCM', 'KIRC', 'GBM', 'PAAD', 'UVM']:
        try:
            print('%-5s  -log10P = %6.3f   enrichment = %d'
                  % (c, log_rank_test(c), clinical_parameters_enrichment_analysis(c)))
        except Exception as e:
            print('%-5s  ERROR: %s' % (c, e))
