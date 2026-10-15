import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from os.path import splitext, basename, isfile
import pandas as pd

from paths import DATASETS_DIR, CLINICAL_DIR, INPUT_DIR

# 数据读取与预处理
def prepare_data(data_folder, cuda, view, noise_intensities):
    # data_folder 癌症类型
    num_view = view
    tmp_dir = os.path.join(DATASETS_DIR, data_folder) + os.sep  # ./datasets/data_folder/
    cli_dir = os.path.join(CLINICAL_DIR, data_folder + '.clinic')
    file_input = os.path.join(INPUT_DIR, data_folder + '.list')
    data_list = []
    sample_name = []

    # 读取初始数据
    for line in open(file_input, 'rt'):  # 遍历fea/cancer_type 文件夹
        base_file = splitext(basename(line.rstrip()))[0]
        # print(base_file)
        fea_save_file = tmp_dir + base_file + '.fea'
        # print(fea_save_file)
        df = pd.read_csv(fea_save_file, sep=',', header=0, index_col=0)
        sample_name = list(df)
        df =df.T
        df_normalized = (df - df.mean()) / df.std()
        # print(data_std)
        # print("无噪音：", df)
        noise = np.random.normal(
            loc=0,  # 均值
            scale=noise_intensities,  # 标准差
            size=df.shape)
        # print("噪音：", noise)
        df = df + noise
        # print("加噪音：", df)
        data_list.append(df.values.astype(float))

    data_tensor_list = []  # 将上述连结后的数据集转化为张量
    data_tensor_list_cpu = []
    for i in range(len(data_list)):
        data_tensor_list.append(torch.FloatTensor(data_list[i]))  # 将上述连结后的数据集转化为张量
        data_tensor_list_cpu.append(torch.FloatTensor(data_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()  # 将上述转化后的张量传入GPU

    # 临床数据
    clinical_data = pd.read_csv(cli_dir)

    return data_tensor_list, sample_name, clinical_data, data_tensor_list_cpu
# data_train_list：张量类型的训练集副本
# data_test_list：张量类型的测试集副本
# idx_dict：字典，包含训练集和测试集的索引
# labels：总标签（训练集+测试集）


class CancerDataset(Dataset):
    # 初始化
    def __init__(self, args):
        self.data_list, self.sample_name, self.clinical_data, self.data_list_cpu = prepare_data(args.data_folder, args.cuda, args.num_view, args.noise_intensities)

    # 计算批量样本的数量
    def __len__(self):  # len()会自动调用该方法
        return len(self.sample_name)

    # 用于将原始数据划分为批量数据
    def __getitem__(self, idx):
        curData = [temp[idx] for temp in self.data_list]
        return curData, torch.from_numpy(np.array(idx))
