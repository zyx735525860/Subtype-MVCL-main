# -*- coding: utf-8 -*-
"""把论文第 3 章的全部表格导出为独立的 Excel 文件（每张表一个 .xlsx）。

表 3.1 直接从仓库数据集统计得到（样本量 / 组学特征维度）；
表 3.2 ~ 3.7 的数值取自论文正文与表格，逐格转录。

输出目录：outputs/tables/
"""
import io
import os
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from paths import DATASETS_DIR, INPUT_DIR, OUTPUT_DIR, ensure_dir

TABLE_DIR = os.path.join(OUTPUT_DIR, 'tables')
CANCERS = ['STAD', 'LUAD', 'UCEC', 'BLCA', 'BRCA',
           'SKCM', 'KIRC', 'GBM', 'PAAD', 'UVM']


def write(df, filename, sheet='Sheet1'):
    ensure_dir(TABLE_DIR)
    path = os.path.join(TABLE_DIR, filename)
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        df.to_excel(w, sheet_name=sheet, index=False)
    print('  [OK] %-46s  %d 行 x %d 列' % (filename, df.shape[0], df.shape[1]))


# ============================================================================
# 表 3.1  10 种 TCGA 数据集的样本量与组学特征维度统计
# ============================================================================
def table31():
    rows = []
    for c in CANCERS:
        # 读取 input/<CANCER>.list 得到该癌种使用的组学文件顺序
        list_file = os.path.join(INPUT_DIR, c + '.list')
        dims = {}
        with open(list_file, 'rt') as f:
            for line in f:
                name = os.path.splitext(os.path.basename(line.strip()))[0]
                fp = os.path.join(DATASETS_DIR, c, name + '.fea')
                if not os.path.exists(fp):
                    continue
                # .fea 的行=组学特征，列=样本。只读表头拿列数，避免加载整个矩阵
                with open(fp, 'rt') as g:
                    n_samples = len(g.readline().split(',')) - 1
                with open(fp, 'rt') as g:
                    n_features = sum(1 for _ in g) - 1
                dims[name] = (n_features, n_samples)

        def get(*keys):
            """返回该组学的特征维度（行数）。"""
            for k in keys:
                if k in dims:
                    return dims[k][0]
            return None

        n_samples = next(iter(dims.values()))[1] if dims else None
        rows.append({
            'Cohort': c,
            'CNV': get('CN', 'CNV'),
            'Meth': get('meth', 'Meth'),
            'mRNA': get('rna', 'mRNA'),
            'miRNA': get('miRNA'),
            'Samples': n_samples,
        })
    df = pd.DataFrame(rows)
    write(df, 'Table3.1_datasets_samples_and_feature_dims.xlsx')
    return df


# ============================================================================
# 表 3.2  模块消融实验 —— 生存分析显著性 (-log10 P)
# ============================================================================
def table32():
    data = {
        'Subtype-MVCL': [1.9, 3.8, 8.3, 3.9, 1.4, 2.9, 9.0, 3.4, 3.6, 2.4],
        'NA':           [0.7, 2.4, 5.3, 0.3, 0.3, 0.1, 4.9, 2.5, 2.6, 2.2],
        'STAR-MLP':     [0.3, 1.9, 6.9, 0.5, 0.4, 3.8, 4.4, 2.2, 2.8, 2.1],
        'MCL':          [0.5, 2.0, 6.9, 3.1, 1.6, 1.9, 6.3, 2.3, 2.5, 2.0],
        'ATT':          [0.4, 1.6, 3.6, 0.1, 0.7, 1.6, 6.7, 2.2, 2.3, 2.4],
        'STAR-MLP+MCL': [0.5, 3.0, 8.2, 2.7, 2.0, 4.8, 7.3, 2.6, 3.5, 2.3],
        'STAR-MLP+ATT': [0.4, 2.6, 6.9, 2.4, 1.5, 5.7, 5.7, 2.3, 2.6, 2.3],
        'MCL+ATT':      [0.5, 2.8, 7.5, 3.2, 2.2, 4.0, 5.6, 3.0, 3.3, 2.4],
    }
    df = pd.DataFrame(data, index=CANCERS).T.reset_index().rename(columns={'index': 'Model'})
    df['Mean'] = df[CANCERS].mean(axis=1).round(2)
    write(df, 'Table3.2_ablation_survival_log10P.xlsx')


# ============================================================================
# 表 3.3  模块消融实验 —— 临床参数显著富集数量
# ============================================================================
def table33():
    data = {
        'Subtype-MVCL': [2, 5, 1, 6, 6, 4, 6, 2, 4, 2],
        'NA':           [2, 2, 1, 3, 6, 2, 5, 1, 0, 0],
        'STAR-MLP':     [2, 3, 1, 2, 5, 2, 5, 1, 0, 1],
        'MCL':          [1, 2, 1, 5, 5, 2, 6, 1, 0, 1],
        'ATT':          [3, 1, 1, 3, 5, 1, 5, 1, 2, 0],
        'STAR-MLP+MCL': [2, 2, 1, 3, 6, 4, 6, 1, 1, 0],
        'STAR-MLP+ATT': [3, 2, 1, 4, 6, 3, 5, 1, 0, 0],
        'MCL+ATT':      [2, 4, 1, 2, 6, 3, 6, 1, 1, 1],
    }
    df = pd.DataFrame(data, index=CANCERS).T.reset_index().rename(columns={'index': 'Model'})
    df['Mean'] = df[CANCERS].mean(axis=1).round(2)
    write(df, 'Table3.3_ablation_clinical_enrichment.xlsx')


# ============================================================================
# 表 3.4  不同方法的生存分析显著性 (-log10 P)
# ============================================================================
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


def table34():
    df = pd.DataFrame(BENCH_SURV, index=CANCERS).T.reset_index().rename(columns={'index': 'Model'})
    df['Mean'] = df[CANCERS].mean(axis=1).round(2)
    write(df, 'Table3.4_benchmark_survival_log10P.xlsx')


def table35():
    df = pd.DataFrame(BENCH_CLIN, index=CANCERS).T.reset_index().rename(columns={'index': 'Model'})
    df['Mean'] = df[CANCERS].mean(axis=1).round(2)
    write(df, 'Table3.5_benchmark_clinical_enrichment.xlsx')


# ============================================================================
# 表 3.6  积分梯度识别的 Top-5 关键生物标志物
# ============================================================================
def table36():
    bio = {
        'STAD': ['CADM3', 'DACT2', 'C7', 'PCSK1N', 'MAMDC2'],
        'LUAD': ['HPGDS', 'GPR110', 'GLB1L3', 'CST6', 'KCNQ5'],
        'UCEC': ['C4BPA', 'C9orf152', 'CDKN2A', 'PNMA3', 'C1orf173'],
        'BLCA': ['KLHDC7A', 'SLC30A2', 'FABP6', 'FER1L4', 'C20orf151'],
        'BRCA': ['C1orf64', 'THSD4', 'FOXA1', 'ROPN1', 'TFF3'],
        'SKCM': ['TFAP2A', 'KIAA0748', 'KCNH1', 'C3orf14', 'ADAMDEC1'],
        'KIRC': ['CA4', 'SCGN', 'GFPT2', 'BTNL9', 'TMEM150C'],
        'GBM':  ['PLCXD3', 'ACSS3', 'FBXO17', 'MMP16', 'HNF4G'],
        'PAAD': ['TRIM29', 'MAGED4B', 'RHOD', 'CXCL5', 'EDA'],
        'UVM':  ['DLL4', 'CCBP2', 'PPP1R3C', 'MTUS1', 'EFS'],
    }
    rows = []
    for c in CANCERS:
        for i, g in enumerate(bio[c], 1):
            rows.append({'Cohort': c, 'Rank': i, 'Biomarker': g})
    write(pd.DataFrame(rows), 'Table3.6_top5_biomarkers_IG.xlsx')

    # 宽表版本更便于阅读
    wide = pd.DataFrame({c: bio[c] for c in CANCERS},
                        index=['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5'])
    wide = wide.reset_index().rename(columns={'index': 'Rank'})
    write(wide, 'Table3.6b_top5_biomarkers_wide.xlsx')


# ============================================================================
# 表 3.7  BRCA 亚型与 PAM50 临床分型的重叠富集
# ============================================================================
def table37():
    counts = {
        'Basal':  [127, 1, 0, 0, 0],
        'Her2':   [15, 48, 2, 1, 0],
        'LumA':   [0, 52, 85, 156, 112],
        'LumB':   [1, 58, 93, 26, 4],
        'Normal': [9, 1, 1, 1, 10],
    }
    clusters = ['C0', 'C1', 'C2', 'C3', 'C4']
    sizes = [152, 160, 181, 184, 126]
    pam50_n = {'Basal': 128, 'Her2': 66, 'LumA': 405, 'LumB': 182, 'Normal': 22}

    M = pd.DataFrame(counts, index=clusters).T          # 行=PAM50，列=簇
    M['Total'] = M.sum(axis=1)

    # 计数表
    cnt = M.reset_index().rename(columns={'index': 'PAM50'})
    write(cnt, 'Table3.7_brca_pam50_overlap_counts.xlsx')

    # 列百分比表（论文正文使用的口径：每个簇内部的构成比）
    pct = (M.drop(columns='Total').div(pd.Series(sizes, index=clusters), axis=1) * 100).round(1)
    pct = pct.reset_index().rename(columns={'index': 'PAM50'})
    for c in clusters:
        pct[c] = pct[c].map(lambda v: '%.1f%%' % v)
    write(pct, 'Table3.7b_brca_pam50_overlap_percent.xlsx')

    # 行百分比表（每个 PAM50 类型落到各簇的比例）
    rpct = (M.drop(columns='Total')
            .div(pd.Series(pam50_n), axis=0) * 100).round(1)
    rpct = rpct.reset_index().rename(columns={'index': 'PAM50'})
    for c in clusters:
        rpct[c] = rpct[c].map(lambda v: '%.1f%%' % v)
    write(rpct, 'Table3.7c_brca_pam50_row_percent.xlsx')


# ============================================================================
# 附加：论文报告值 vs 本仓库已有结果的重算值
# ============================================================================
def table_extra():
    from stats_py import log_rank_test, clinical_parameters_enrichment_analysis
    paper_s = {'STAD': 1.9, 'LUAD': 3.8, 'UCEC': 8.3, 'BLCA': 3.9, 'BRCA': 1.4,
               'SKCM': 2.9, 'KIRC': 9.0, 'GBM': 3.4, 'PAAD': 3.6, 'UVM': 2.4}
    paper_c = {'STAD': 2, 'LUAD': 5, 'UCEC': 1, 'BLCA': 6, 'BRCA': 6,
               'SKCM': 4, 'KIRC': 6, 'GBM': 2, 'PAAD': 4, 'UVM': 2}
    rows = []
    for c in CANCERS:
        rows.append({
            'Cohort': c,
            'Paper_log10P': paper_s[c],
            'Repro_log10P': round(float(log_rank_test(c)), 3),
            'Paper_Clinical': paper_c[c],
            'Repro_Clinical': int(clinical_parameters_enrichment_analysis(c)),
        })
    df = pd.DataFrame(rows)
    df['AbsDiff_log10P'] = (df['Repro_log10P'] - df['Paper_log10P']).abs().round(3)
    write(df, 'TableX_reproduction_comparison.xlsx')


if __name__ == '__main__':
    print('=' * 74)
    print('导出论文表格 -> %s' % TABLE_DIR)
    print('=' * 74)
    table31()
    table32()
    table33()
    table34()
    table35()
    table36()
    table37()
    print()
    print('--- 附加表（复现对照，非论文原表）---')
    table_extra()
    print()
    print('完成。共 %d 个 Excel 文件。' % len(os.listdir(TABLE_DIR)))
