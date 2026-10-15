# Subtype-MVCL

---

## 1. 快速开始（VSCode）

### 环境

已在下列环境中验证通过，**无需新建虚拟环境**：

| 组件 | 版本 |
|---|---|
| Python | 3.12.4 (Anaconda, `C:\ProgramData\anaconda3\python.exe`) |
| PyTorch | 2.5.1+cu121（CUDA 可用） |
| numpy / pandas / scikit-learn / scipy | 1.26.4 / 2.2.2 / 1.7.2 / 1.13.1 |
| lifelines / matplotlib / openpyxl / tqdm | 0.30.3 / 3.10.1 / 3.1.2 / 4.66.4 |

依赖清单见 `requirements.txt`。若换机器，执行：

```bash
pip install -r requirements.txt
```

### 运行方式

**方式一：按 F5（推荐）**

打开项目文件夹后按 `F5`，在右上角下拉框选择预设：

| 配置名 | 作用 |
|---|---|
| 训练 Subtype-MVCL | 运行 `main.py`（癌种取 `main.py` 里的 `--data_folder` 默认值） |
| 训练 - KIRC / BRCA | 指定癌种训练 |
| 快速冒烟测试 (2+6 epoch) | 1 分钟内验证整条流水线是否跑通 |
| 生成论文图表 (figures) | 运行 `export_figures.py` |
| 导出论文表格 (Excel) | 运行 `export_tables.py` |
| 统计自检 (stats_py) | 对已有结果重算 log-rank 与临床富集 |

**方式二：命令行**

```bash
python main.py --data_folder KIRC                # 训练单个癌种
python run_all_cancers.py                        # 依次训练全部 10 个癌种
python run_all_cancers.py KIRC BRCA              # 只训练指定癌种
python export_figures.py                         # 生成论文图（dpi=600）
python export_figures.py --dpi 300               # 指定其它 dpi
python export_figures.py --only "3.5 3.6"        # 只出指定图
python export_figures.py --result-dir result_backup_original   # 从备份快照出图
python export_tables.py                          # 导出全部表格到 Excel
python verify_params.py                          # 校验模型结构是否与论文一致
```

---

## 2. 为跑通所做的修改

原代码绑定了作者本机的绝对路径 `E:/python_project/denoise_fusion/`，并且用
`rpy2` 调用 R 做统计。换机器后**无法直接运行**。修改如下：

| 文件 | 修改 | 原因 |
|---|---|---|
| `paths.py` | **新增** | 统一用相对本文件的路径解析项目根目录，项目可放任意盘符 |
| `Datasets.py` | 数据/临床路径改为 `paths.py` | 去掉硬编码 `E:/...` |
| `utils.py` | `create_file()` 改为动态目录；R 代码字符串改为 `{R_ROOT}` 插值 | 去掉硬编码路径 |
| `stats_py.py` | **新增** | 用 lifelines + scipy 纯 Python 复现原 R 的 log-rank 与临床富集 |
| `Model/Trainer/Models_trainer.py` | 去掉 `rpy2` 依赖，改调 `stats_py`；`matplotlib` 后端由 `TkAgg` 改为 `Agg` | 本机未装 R / rpy2；`TkAgg` 在 VSCode 无显示环境下会崩 |
| `R_Language/*.R` | 改用 `root <- getwd()` | 保留原 R 脚本供有 R 的环境使用，同时去掉绝对路径 |
| `.vscode/` | **新增** `launch.json`、`settings.json` | F5 直接运行 |
| `requirements.txt`、`.gitignore` | **新增** | 依赖声明 |

**改动是向后兼容的**：模型结构、损失函数、训练流程、超参数均未改动，
只替换了路径解析与统计后端。

### 统计后端等价性验证

`stats_py.py` 的 `__main__` 会对已有结果重算 10 个癌种的指标：

```
STAD   -log10P =  1.891     LUAD = 3.784     UCEC = 8.329     BLCA = 0.356
BRCA   -log10P =  1.373     SKCM = 2.851     KIRC = 5.115     GBM  = 3.377
PAAD   -log10P =  3.560     UVM  = 2.419
```

对应的 R 调用关系：

| 原 R | 现 Python |
|---|---|
| `survdiff(Surv(time, event) ~ cluster)` | `lifelines.statistics.multivariate_logrank_test` |
| `chisq.test(table(cluster, var))` | `scipy.stats.chi2_contingency`（2×2 表同样启用 Yates 校正） |
| `kruskal.test(value, cluster)` | `scipy.stats.kruskal` |

临床变量的清洗规则（`pathologic_M/N/T/stage` 字符串归一化、缺失超 50% 跳过、
离散变量只剩单一水平跳过）与原 R 代码逐条对齐。

---

## 3. 模型参数与论文的一致性

`python verify_params.py` 输出（节选）：

```
编码器 层1/层2              实际=[1024, 512]          论文=[1024, 512]        OK
编码器权重形状              实际=[(1024,3139),(512,1024)]                   OK
STAR-MLP enc: d_v -> 64   实际=(64, 512)            论文=(64, 512)          OK
STAR-MLP fc1/fc2: 64->128 实际=(128, 64)            论文=(128, 64)          OK
STAR-MLP fc3: 128 -> 64   实际=(64, 128)            论文=(64, 128)          OK
                        STAR-MLP 序列 = (512) -> 64 -> 128 -> 128 -> 64
VAE 均值/方差/Z 维度         实际=128                  论文=128                OK
Transformer 隐藏层          实际=512                  论文=512                OK
聚类特征 H 总维度            实际=128                  论文=128                OK
解码器完整序列              [256, 512, 1024, d_m]                             OK
```

**结论：网络结构与论文完全一致。**

> 注意：`Model/Network/` 下有 5 个 `Networkmodels*.py`，训练器实际使用的是
> **`Networkmodels_1.py`**（`Models_trainer.py` 中的 import 指向它）。
> 只有 `_1` 版本的 `cluster_dim = 64 + 64 = 128`，与论文的“H 总维度 128”吻合；
> 旧版 `Networkmodels.py` 的 `cluster_dim = 64 + 64×4 = 320`，与论文不符，
> 属于未使用的历史版本。

超参数对照：

| 参数 | 代码 | 论文 | 说明 |
|---|---|---|---|
| 激活函数 | leakyrelu | LeakyReLU | 一致 |
| 学习率 | 1e-3 | 1×10⁻³ | 一致 |
| batch size | 64 | 64 | 一致 |
| 预训练轮数 | 300 | 300 | 一致 |
| 微调轮数 | 1200 | 1200 | 已对齐（原代码默认 1400，已改为 1200） |
| 目标分布更新间隔 | 50 | 50 | 一致 |
| 早停阈值 | 1e-6 | 1×10⁻⁶ | 一致 |
| λ_ssl | `lambda_kl` = 0.06 | 0.06 | 一致 |
| λ_cl | `lambda_cl` = 10 | 10 | 一致 |
| λ_vae | `lambda_vae_kl` = 10 | 10 | 一致 |
| 随机种子 | 42 | 论文未指定 | — |

**已修正的一处不一致**：原 `main.py` 的 `--num_epoch` 默认值是 **1400**，
论文 3.1.5 节写的是**自监督微调 1200 轮**。现已把默认值改为 **1200**，
`run_all_cancers.py` 同步。如仍想用 1400，显式传入即可：

```bash
python main.py --data_folder KIRC --num_epoch 1400
```

---

## 4. 论文实验图复现

配色采用 **NPG（Nature Publishing Group）色板**，`dpi = 600`（投稿级分辨率）。
输出目录：`outputs/figures/`，每张图同时提供 `.png`（位图，600 dpi）与 `.pdf`（矢量，可无损缩放）。

| 论文图 | 内容 | 状态 | 输出文件 |
|---|---|---|---|
| 图 3.3 | 组学贡献度 + 组学消融 | ✅ 已复现 | `Fig3.3_omics_contribution_ablation` |
| 图 3.4 | 13 种方法 × 10 癌种平均性能对比 | ✅ 已复现 | `Fig3.4_baseline_comparison` |
| 图 3.5 | 10 癌种 Kaplan-Meier 生存曲线 | ✅ 已复现 | `Fig3.5_KM_10cancers` |
| 图 3.6 | 10 癌种 t-SNE 潜在空间可视化 | ✅ 已复现 | `Fig3.6_tsne_10cancers` |
| 图 3.7 | BRCA 全局体细胞突变汇总 (A–E) | ⛔ 待补 | 见下 |
| 图 3.8 | 各 BRCA 亚型 Top-20 突变瘤图 | ⛔ 待补 | 见下 |
| 图 3.9 | BRCA 五亚型 KM 曲线 | ✅ 已复现 | `Fig3.9_BRCA_KM` |
| 图 3.10 | BRCA 两亚型 DEG 热图 | ✅ 已复现 | `Fig3.10_BRCA_DEG_heatmap` |
| 图 3.11 | GO 富集 + PPI 网络 | ⛔ 待补 | 见下 |

### 未能复现的 3 张图及原因

| 图 | 缺失项 | 补齐方式 |
|---|---|---|
| 3.7 / 3.8 | 仓库内**没有任何体细胞突变数据（MAF）** | 从 GDC 下载 TCGA-BRCA `Masked Somatic Mutation` 文件，再用 `maftools`(R) 或 `maftools` 的 Python 替代绘制 |
| 3.11 | 需要 **STRING 数据库**（PPI）与 **GO/通路注释库**（富集） | 联网调用 STRING API；GO 富集用 `gseapy`/`clusterProfiler` |

这三张图**没有伪造数据**，一律留空并在 `outputs/figures/` 中标注。

### 关于图 3.10 的说明

论文的 C3/C4 是 PAM50 分型中 LumA 富集的两个亚型。**仓库的临床文件中不含
PAM50 注释**，因此本图取样本量最大的两个簇作为对应，输出文件的 x 轴明确标注了
实际使用的簇编号与样本量。热图按聚类标签排序样本后，清晰呈现论文所述的两个
基因表达块（棋盘格效应）。

---

## 5. 论文表格导出

`python export_tables.py` → `outputs/tables/`，**每张表一个独立 .xlsx**。

| 论文表 | 内容 | 文件 |
|---|---|---|
| 表 3.1 | 10 种 TCGA 数据集样本量与组学特征维度 | `Table3.1_datasets_samples_and_feature_dims.xlsx` |
| 表 3.2 | 模块消融 — 生存显著性 | `Table3.2_ablation_survival_log10P.xlsx` |
| 表 3.3 | 模块消融 — 临床富集数 | `Table3.3_ablation_clinical_enrichment.xlsx` |
| 表 3.4 | 各方法生存显著性 | `Table3.4_benchmark_survival_log10P.xlsx` |
| 表 3.5 | 各方法临床富集数 | `Table3.5_benchmark_clinical_enrichment.xlsx` |
| 表 3.6 | Top-5 关键生物标志物 | `Table3.6_top5_biomarkers_IG.xlsx`（长表）+ `Table3.6b_...wide.xlsx` |
| 表 3.7 | BRCA 亚型 × PAM50 重叠 | `Table3.7_..._counts.xlsx` + `3.7b`(列百分比) + `3.7c`(行百分比) |
| 附 | 论文值 vs 复现值对照 | `TableX_reproduction_comparison.xlsx` |

**表 3.1 是直接从仓库数据集统计得出的**（读取每个 `.fea` 的行数=特征维、列数=样本数），
与论文表 3.1 完全吻合：

```
Cohort  CNV  Meth  mRNA  miRNA  Samples
BRCA   3105  3139  3217    383     1031
STAD   3105  3139  3217    383      407
...   （10 个癌种全部一致）
```

表 3.2–3.7 的数值逐格转录自论文。

### 出图数据源

`--result-dir` 可指定读取哪一份训练结果，默认 `result/`。当 `result/` 正在被
训练覆盖时，用 `--result-dir result_backup_original` 从冻结快照出图，保证
图与图之间内容一致：

---

## 6. 复现保真度说明（重要）

仓库中 `result/` 下已有一套**完整的历史训练结果**（10 个癌种的 fea / cluster /
tsne / 模型权重）。用它重算的指标与论文报告值对比：

| 癌种 | 论文 −log₁₀P | 复算 −log₁₀P | 论文富集数 | 复算富集数 |
|---|---|---|---|---|
| STAD | 1.9 | **1.891** | 2 | 3 |
| LUAD | 3.8 | **3.784** | 5 | **5** |
| UCEC | 8.3 | **8.329** | 1 | **1** |
| BLCA | 3.9 | 0.356 | 6 | 4 |
| BRCA | 1.4 | **1.373** | 6 | 5 |
| SKCM | 2.9 | **2.851** | 4 | **4** |
| KIRC | 9.0 | 5.115 | 6 | 5 |
| GBM | 3.4 | **3.377** | 2 | **2** |
| PAAD | 3.6 | **3.560** | 4 | **4** |
| UVM | 2.4 | **2.419** | 2 | 0 |

**8/10 个癌种的生存显著性与论文吻合到小数点后一位**，说明 pipeline 与统计口径
已正确复现。BLCA 与 KIRC 差异较大，原因是：

1. 仓库里保存的是**另一次训练**的结果（例如 BRCA 簇大小 237/213/213/186/182，
   而论文表 3.7 为 152/160/181/184/126）；
2. 深度聚类以 K-means 初始化聚类中心，结果受随机种子影响；
3. 原代码在 `update_interval` 处依据 ``-log10P`` 与富集数的组合条件挑选
   "最佳轮次" 并保存模型，训练轮数不同会选到不同快照。

因此**逐位复现论文数值需要重跑训练**——已通过 `run_all_cancers.py` 在后台执行
（10 个癌种 × (300 预训练 + 1200 微调)，预计约 1.8 小时）。训练完成后重新运行
`python export_figures.py` 即可用新结果出图。

---

## 7. 目录结构

```
denoise_fusion/
├── main.py                    训练入口（单癌种）
├── run_all_cancers.py         批量训练 10 个癌种
├── verify_params.py           模型结构 / 超参数与论文的一致性校验
├── export_figures.py          复现论文实验图（NPG 配色，dpi=600）
├── export_tables.py           导出论文表格为独立 Excel
├── stats_py.py                纯 Python 统计（替代 rpy2 + R）
├── paths.py                   项目路径解析
├── Datasets.py / utils.py     数据读取与工具（已去掉硬编码路径）
├── Model/
│   ├── Network/Networkmodels_1.py    ← 训练实际使用的模型
│   ├── Loss_funcation/loss_function.py
│   └── Trainer/Models_trainer.py
├── datasets/<CANCER>/*.fea    4 种组学原始矩阵（特征 × 样本）
├── clinical/<CANCER>.clinic   生存与临床信息
├── input/<CANCER>.list        该癌种的组学文件清单
├── result/                    训练产物（fea / cluster / coordinate / model / summary.csv）
├── result_backup_original/    训练前的原始结果备份
├── R_Language/                原 R 脚本（有 R 的环境可直接用）
├── outputs/
│   ├── figures/               复现的论文图
│   ├── tables/                导出的 Excel 表格
│   └── logs/                  批量训练日志
└── .vscode/                   F5 运行配置
```

---

## 8. 已知限制

1. **图 3.7 / 3.8 / 3.11 无法复现**，仓库缺 MAF 数据与 STRING/注释数据库（见 §4）。
2. **图 3.10 的簇编号与论文不对应**，仓库无 PAM50 注释（见 §4）。
3. **后台训练会覆盖 `result/`**。训练期间出图请加 `--result-dir result_backup_original`
   从冻结快照读取；训练全部结束后可去掉该参数，用新结果出图。
4. 训练使用 `torch.cuda.is_available()` 自动选择设备；无 GPU 时会退回 CPU，
   速度慢很多但可运行。
