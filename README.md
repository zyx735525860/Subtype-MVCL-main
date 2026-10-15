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



## 2. 目录结构

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
