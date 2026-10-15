# -*- coding: utf-8 -*-
"""项目根路径解析。

原代码把 `E:/python_project/denoise_fusion/` 硬编码在 Datasets.py / utils.py /
R_Language 中，换机器就找不到文件。这里统一用相对本文件的路径解析，
项目可以放在任意盘符、任意目录下直接运行。
"""
import os

# 本文件所在目录即项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASETS_DIR = os.path.join(PROJECT_ROOT, 'datasets')
CLINICAL_DIR = os.path.join(PROJECT_ROOT, 'clinical')
INPUT_DIR = os.path.join(PROJECT_ROOT, 'input')
RESULT_DIR = os.path.join(PROJECT_ROOT, 'result')

RESULT_FEA_DIR = os.path.join(RESULT_DIR, 'fea')
RESULT_CLUSTER_DIR = os.path.join(RESULT_DIR, 'cluster')
RESULT_COORD_DIR = os.path.join(RESULT_DIR, 'coordinate')
RESULT_MODEL_DIR = os.path.join(RESULT_DIR, 'model')
RESULT_PRE_MODEL_DIR = os.path.join(RESULT_MODEL_DIR, 'pre_train_model')
RESULT_TRAINED_MODEL_DIR = os.path.join(RESULT_MODEL_DIR, 'trained_model')
RESULT_SUMMARY_CSV = os.path.join(RESULT_DIR, 'summary.csv')

# 输出目录（图表 / Excel）
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')


def ensure_dir(path):
    """确保目录存在并返回该路径。"""
    os.makedirs(path, exist_ok=True)
    return path


def r_literal(path):
    """把 Windows 路径转成 R 字符串字面量可用的正斜杠形式。"""
    return path.replace('\\', '/')
