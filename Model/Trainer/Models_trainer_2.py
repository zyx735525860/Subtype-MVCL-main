from tqdm import tqdm
import torch
import random
import numpy as np
import pandas as pd
import os
from os.path import isfile
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import rpy2.robjects
import matplotlib
os.environ["OMP_NUM_THREADS"] = '5'
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from Datasets import CancerDataset
from Model.Network.Networkmodels_1 import network
from Model.Loss_funcation.loss_function import loss
from utils import log_rank_test, clinical_parameters_enrichment_analysis, probability_distribution_P, probability_distribution_Q, create_file

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# 随机种子
def set_seed(seed=0):
    # credits to CC and https://pytorch.org/docs/stable/notes/randomness.html
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    # 设置PyTorch的随机种子，用于控制CPU上生成的随机数
    torch.cuda.manual_seed_all(seed)
    # 设置PyTorch所有可见的CUDA设备的随机种子
    torch.cuda.manual_seed(seed)
    # 设置当前CUDA设备的随机种子
    np.random.seed(seed)
    # 设置NumPy库的随机种子
    random.seed(seed)
    # 设置Python内置的random模块的随机种子

class model_trainer(object):
    # 初始化
    def __init__(self, args, fix_seed=True):
        self.args = args
        if fix_seed:
            set_seed(self.args.random_seed)

        self.dataset = CancerDataset(self.args)
        self.args.dim_list = [x.shape[1] for x in self.dataset.data_list]
        # print(self.args.dim_list)
        # print(self.args)

        self.model = network(self.args)
        self.loss = loss(self.args)
        # 创建文件
        self.result_fea_file, self.result_cluster_file, self.result_pre_model_file, self.result_model_file, self.result_csv_file = create_file(
            self.args.data_folder)
        # print(self.model)

    # 模型训练
    def train(self, need_pre_train=True):
        self.model.cuda()
        optimizer = Adam(self.model.parameters(), lr=self.args.lr)

        # 优化器
        if self.args.scheduler == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.args.pre_epoch + self.args.num_epoch, eta_min=self.args.min_lr)
        elif self.args.scheduler == "get_cosine_schedule_with_warmup":
            num_training_steps = self.args.pre_epoch + self.args.num_epoch
            num_warmup_steps = int(0.1 * num_training_steps)
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

        if (need_pre_train):
            print("===================")
            print("pre_train-dataset-: {} cluster_num-: {}".format(self.args.data_folder, self.args.cluster_number))

            pbar = tqdm(range(self.args.pre_epoch), ncols=140)
            for _ in pbar:
                train_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
                self.model.train()

                # 损失初始化
                total_loss = 0.0  # 初始化累积损失
                total_recon_loss = 0.0  # 重构损失
                total_vae_kl_loss = 0.0  # VAE的KL损失
                total_cl_loss = 0.0 # 对比损失
                num_batches = 0  # 初始化批次数

                # 批量训练
                for batch_idx, (x, idx) in enumerate(train_loader):
                    x = [x_dat.to(self.args.device) for x_dat in x]
                    batch_size = len(idx)
                    list_dict = self.model(x)

                    # 计算损失
                    recon_loss = self.loss.Reconstruction_loss(dec_list=list_dict["recon_list"], x=x, num=batch_size)
                    vae_kl_loss = self.loss.Vae_kl_loss(mu=list_dict["mu_list"], logvar=list_dict["logvar_list"], num=batch_size)
                    cl_loss = self.loss.ContrastiveLoss(share_list=list_dict["share_list"], noise_list=list_dict["noise_list"], unique_list=list_dict["unique_list"], batch_size=batch_size)
                    loss = recon_loss + vae_kl_loss + cl_loss

                    # 梯度更新
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.001)
                    optimizer.step()

                    # 损失计算
                    total_loss += loss.item()
                    total_recon_loss += recon_loss.item()
                    total_vae_kl_loss += vae_kl_loss.item()
                    total_cl_loss += cl_loss.item()
                    num_batches += 1

                # 一轮次损失可视化
                avg_loss = total_loss / num_batches
                avg_recon_loss = total_recon_loss / num_batches
                avg_vae_kl_loss = total_vae_kl_loss / num_batches
                avg_cl_loss = total_cl_loss / num_batches
                pbar.set_postfix(ALL_Loss="{:3.3f}".format(avg_loss),
                                 Recon_Loss="{:3.3f}".format(avg_recon_loss),
                                 VAE_Loss="{:.3e}".format(avg_vae_kl_loss),
                                 CL_Loss="{:.3e}".format(avg_cl_loss)
                                 )
                scheduler.step()

        # 存储(读取)模型
        if (need_pre_train):
            self.model.cpu()
            state = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(state, self.result_pre_model_file)
            print("pre_training finished")
            self.model.cuda()
        else:
            checkpoint = torch.load(self.result_pre_model_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# =======================training==================================
# ========================Initialize cluster centers via K-means=====================
        self.model.eval()
        list_dict = self.model(self.dataset.data_list)
        print("train-dataset-: {} cluster_num-: {}".format(self.args.data_folder, self.args.cluster_number))
        # print(list_dict["cluster_data_list"])
        kmeans = KMeans(self.args.cluster_number, random_state=self.args.random_seed, init="k-means++", n_init=20).fit(
                        list_dict["cluster_data_list"][0].detach().cpu().numpy())
        # with torch.no_grad():
        self.model.cluster_layer.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))

        y_pred_last = kmeans.labels_
        max_label_log = 0.0
        max_label_enrichment_num = 0
        max_label_pred_p = y_pred_last
        pbar = tqdm(range(self.args.num_epoch), ncols=140)
        for epoch in pbar:
            if epoch % self.args.update_interval == 0:
                with torch.no_grad():
                    self.model.eval()
                    list_dict = self.model(self.dataset.data_list)
                    cluster_feat = list_dict["cluster_data_list"][0]
                    cluster_feat_cpu = cluster_feat.cpu().numpy()
                    df = pd.DataFrame(data=cluster_feat_cpu, index=self.dataset.sample_name,
                                      columns=map(lambda feat: 'v' + str(feat), range(cluster_feat_cpu.shape[1])))
                    df.to_csv(self.result_fea_file, header=True, index=True, sep='\t')

                    q = probability_distribution_Q(cluster_feat=cluster_feat, cluster_centers=self.model.cluster_layer,
                                                   alpha=self.args.alpha, device=cluster_feat.device)
                    p = probability_distribution_P(q)

                    y_pred = q.cpu().numpy().argmax(1)
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = y_pred
                    df['Labels'] = y_pred
                    df = df.loc[:, ['Labels']]
                    df.to_csv(self.result_cluster_file, header=True, index=True, sep='\t')

                    p_code = log_rank_test(self.args.data_folder)
                    p_value = torch.tensor(rpy2.robjects.r(p_code))
                    log10_p = -torch.log10(p_value)

                    e_code = clinical_parameters_enrichment_analysis(self.args.data_folder)
                    enrichment_num = torch.tensor(rpy2.robjects.r(e_code))
                    print('===========')
                    print("p_value: {:.3f}".format(log10_p.item()))
                    print("enrichment_num:", enrichment_num.item())
                    if log10_p > max_label_log and enrichment_num >= max_label_enrichment_num:
                        max_label_log = log10_p
                        max_label_enrichment_num = enrichment_num
                        print('===========')
                        print("max_p_value: {:.3f}".format(max_label_log.item()))
                        print("max_enrichment_num:", enrichment_num.item())
                        max_label_pred_p = y_pred
                        torch.save(self.model.state_dict(), self.result_model_file)
                    elif log10_p > -torch.log10(torch.tensor(0.05)) and enrichment_num > max_label_enrichment_num:
                        print('===========')
                        print("p_value: {:.3f}".format(log10_p.item()))
                        print("enrichment_num:", enrichment_num.item())
                        max_label_pred_p = y_pred
                        max_label_log = log10_p
                        max_label_enrichment_num = enrichment_num
                        torch.save(self.model.state_dict(), self.result_model_file)

                    if epoch > 0 and delta_label < self.args.stop:
                        print('delta_label {:.4f}'.format(delta_label), '< tol',self.args.stop)
                        print('Reached tolerance threshold. Stopping training.')
                        break

            train_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            self.model.train()
            total_loss = 0.0  # 初始化累积损失
            total_recon_loss = 0.0  # 重构损失
            total_vae_kl_loss = 0.0  # VAE的KL损失
            total_cl_loss = 0.0  # 对比损失
            total_kl_loss = 0.0 # KL散度损失
            num_batches = 0  # 初始化批次数
            for batch_idx, (x, idx) in enumerate(train_loader):
                x = [x_dat.to(self.args.device) for x_dat in x]
                batch_size = len(idx)
                list_dict = self.model(x)

                q = probability_distribution_Q(cluster_feat=list_dict["cluster_data_list"][0], cluster_centers=self.model.cluster_layer,
                                               alpha=self.args.alpha, device=list_dict["cluster_data_list"][0].device)
                kl_loss = self.loss.Kl_loss(q=q, p=p[idx])
                recon_loss = self.loss.Reconstruction_loss(dec_list=list_dict["recon_list"], x=x, num=batch_size)
                vae_kl_loss = self.loss.Vae_kl_loss(mu=list_dict["mu_list"], logvar=list_dict["logvar_list"],
                                                    num=batch_size)
                cl_loss = self.loss.ContrastiveLoss(share_list=list_dict["share_list"],
                                                    noise_list=list_dict["noise_list"],
                                                    unique_list=list_dict["unique_list"], batch_size=batch_size)
                loss = recon_loss + vae_kl_loss + cl_loss + kl_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.001)
                optimizer.step()

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_vae_kl_loss += vae_kl_loss.item()
                total_cl_loss += cl_loss.item()
                total_kl_loss += kl_loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            avg_recon_loss = total_recon_loss / num_batches
            avg_vae_kl_loss = total_vae_kl_loss / num_batches
            avg_kl_loss = total_kl_loss / num_batches
            avg_cl_loss = total_cl_loss / num_batches
            scheduler.step()

            pbar.set_postfix(ALL_Loss="{:3.3f}".format(avg_loss),
                             Recon_Loss="{:3.3f}".format(avg_recon_loss),
                             VAE_Loss="{:.3e}".format(avg_vae_kl_loss),
                             KL_Loss="{:.3e}".format(avg_kl_loss),
                             CL_Loss="{:.3e}".format(avg_cl_loss)
                            )

        checkpoint = torch.load(self.result_model_file)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        list_dict = self.model(self.dataset.data_list)
        cluster_feat = list_dict["cluster_data_list"][0].detach().cpu()
        df = pd.DataFrame(data=cluster_feat, index=self.dataset.sample_name,
                          columns=map(lambda feat: 'v' + str(feat), range(cluster_feat.shape[1])))
        df.to_csv(self.result_fea_file, header=True, index=True, sep='\t')
        df['Labels'] = max_label_pred_p
        df = df.loc[:, ['Labels']]
        df.to_csv(self.result_cluster_file, header=True, index=True, sep='\t')

        p_code = log_rank_test(self.args.data_folder)
        p_value = torch.tensor(rpy2.robjects.r(p_code))
        log10_p = -torch.log10(p_value)

        e_code = clinical_parameters_enrichment_analysis(self.args.data_folder)
        enrichment_num = torch.tensor(rpy2.robjects.r(e_code))

        if not os.path.isfile(self.result_csv_file):
            target = open(self.result_csv_file, "w")
            target.write(
                "cancer, pre_epoch, dropout, lambda_rec, lambda_kl, lambda_cl, lambda_vae_kl, star_mutl_dim, p_value, enrichment_num\n")
            target.flush()
        target = open(self.result_csv_file, "a")
        target.write(f"{self.args.data_folder}, " \
                     f"{self.args.pre_epoch}, " \
                     f"{self.args.dropout}, " \
                     f"{self.args.lambda_rec}, " \
                     f"{self.args.lambda_kl}, " \
                     f"{self.args.lambda_cl}, " \
                     f"{self.args.lambda_vae_kl}, " \
                     f"{self.args.star_mutl_dim}, " \
                     f"{log10_p.item()}, " \
                     f"{enrichment_num.item()}, " \
                     f"{self.args.head_num}, " \
                     f"{self.args.lr}, " \
                     f"{self.args.batch_size}, " \
                     f"{self.args.scheduler}, " \
                     f"{self.args.act}\n")
        target.flush()

    # 样本聚类可视化
    def show(self):
        cancer_type = self.args.data_folder
        result_fea_file = './result/fea/' + self.args.data_folder + '.fea'  # 整合后的特征文件
        coordinate_file = './result/coordinate/' + cancer_type + '.tsne'  # 坐标文件
        label_file = './result/cluster/' + cancer_type + '.cluster'  # 聚类结果文件

        # 用tsne方法对高维样本进行降维，即变成x，y两个维度用于投射到坐标轴之中
        if isfile(result_fea_file):
            df1 = pd.read_csv(result_fea_file, header=0, index_col=0, sep='\t')
            mat = df1.values.astype(float)
            labels = self.model.tsne(mat)
            # print(labels.shape)
            df1['x'] = labels[:, 0]  # 设置x列
            df1['y'] = labels[:, 1]  # 设置y列
            df1 = df1.loc[:, ['x', 'y']]
            # 从 DataFrame df1 中选择 ‘x’ 和 ‘y’ 两列，并创建一个新的 DataFrame df1，只包含这两列的数据
            df1.to_csv(coordinate_file, header=True, index=True, sep='\t')
            if isfile(label_file):
                df2 = pd.read_csv(label_file, header=0, index_col=0, sep='\t')
                df1 = pd.merge(df1, df2, left_index=True, right_index=True)

                d = df1[df1['Labels'] == 0]
                plt.scatter(d.values[:, 0], d.values[:, 1], c='r', label='c1')

                d = df1[df1['Labels'] == 1]
                plt.scatter(d.values[:, 0], d.values[:, 1], c='y', label='c2')

                d = df1[df1['Labels'] == 2]
                plt.scatter(d.values[:, 0], d.values[:, 1], c='g', label='c3')

                d = df1[df1['Labels'] == 3]
                plt.scatter(d.values[:, 0], d.values[:, 1], c='b', label='c4')

                d = df1[df1['Labels'] == 4]
                plt.scatter(d.values[:, 0], d.values[:, 1], c='c', label='c5')

                plt.xlabel("tSNE Dimension1", fontsize=14)
                plt.ylabel("tSNE Dimension2", fontsize=14)
                plt.legend(['c1', 'c2', 'c3', 'c4', 'c5'], title="Cluster", loc=1,
                            bbox_to_anchor=(1.105, 1.0),
                            borderaxespad=0.)
                # plt.legend(['c1', 'c2', 'c3'], title="Cluster")
                plt.show()