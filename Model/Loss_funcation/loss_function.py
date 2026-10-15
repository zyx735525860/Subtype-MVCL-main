import torch
import torch.nn as nn
import torch.nn.functional as F

class loss(object):
    # 初始化
    def __init__(self, args):
        # 计算正负样本的索引矩阵
        self.args = args
        if self.args.lambda_cl != 0:
            self.similarity_measure = nn.CosineSimilarity(dim=2)

            # self.mask_positives：正样本mark矩阵--同一样本不同视图内的共享数据视为正样本
            # self.mask_share_to_other_share：负样本mark矩阵--不同样本不同视图内的共享数据视为负样本
            # self.mask_share_unique_noise：负样本mark矩阵--每个视图内样本的共享数据与所有视图内样本的噪音数据
            #                                           --每个视图内样本的独有数据与所有视图内样本的噪音数据
            #                                           --每个视图内样本的独有数据与所有视图内样本的共享数据
            # self.mark_unique_to_unique：负样本mark矩阵--每个视图内样本的独有数据与其他视图内所有样本的独有数据
            self.mask_dict = dict()
            self.num_dict = dict()
            self.mask_dict["mask_positives"], self.mask_dict["mask_share_to_other_share"], self.mask_dict[
                "mask_share_unique_noise"], \
            self.mask_dict["mark_unique_to_unique"], self.num_dict["num_share_to_other_share"], self.num_dict[
                "num_positives"], \
            self.num_dict["num_share_unique_noise"], self.num_dict["num_unique_to_unique"] = self.mark_matrix(
                self.args.num_view, self.args.batch_size)
            # print(self.mask_dict)

    # 计算mark矩阵
    def mark_matrix(self, num_view, batch_size):
        mat_dim = num_view * batch_size  # 维度

        # 正样本：同一个样本的不同视图共享数据
        mask_positives = torch.zeros(mat_dim, mat_dim)
        for i in range(num_view):
            for j in range(num_view):
                if i != j:  # 同一个视图内的不互相视正样本
                    mask_positives[((i + 0) * batch_size):((i + 1) * batch_size),
                    ((j + 0) * batch_size):((j + 1) * batch_size)].fill_diagonal_(1)
        mask_positives = mask_positives.bool()

        # 负样本：每个样本的共享与不同视图的其他样本共享数据
        mask_share_to_other_share = ~mask_positives
        for i in range(num_view):
            mask_share_to_other_share[((i + 0) * batch_size):((i + 1) * batch_size),
            ((i + 0) * batch_size):((i + 1) * batch_size)] = False

        # 负样本：
        # 每个共享数据与所有噪音数据，
        # 每个独有数据与所有噪音数据，
        # 每个样本的独有数据与所有视图的所有共享数据
        neg_mask_dim = num_view * batch_size * 2
        mask_share_unique_noise = torch.ones(neg_mask_dim, neg_mask_dim)
        for i in range(2):
            mask_share_unique_noise[((i + 0) * (num_view * batch_size)):((i + 1) * (num_view * batch_size)),
                                    ((i + 0) * (num_view * batch_size)):((i + 1) * (num_view * batch_size))] = False
        mask_share_unique_noise = mask_share_unique_noise.bool()

        # 负样本：每个样本的独有数据与其他视图的所有独有数据
        mark_unique_to_unique = torch.ones(mat_dim, mat_dim).bool()
        for i in range(self.args.num_view):
            mark_unique_to_unique[((i + 0) * batch_size):((i + 1) * batch_size),
                                  ((i + 0) * batch_size):((i + 1) * batch_size)] = False

        # 各类型正负样本的数量
        num_share_to_other_share = torch.sum(mask_share_to_other_share).item()
        num_positives = torch.sum(mask_positives).item()
        num_share_unique_noise = torch.sum(mask_share_unique_noise).item()
        num_unique_to_unique = torch.sum(mark_unique_to_unique).item()
        return mask_positives, mask_share_to_other_share, mask_share_unique_noise, mark_unique_to_unique,\
                num_share_to_other_share, num_positives, num_share_unique_noise, num_unique_to_unique

    # 计算正样本得分
    def Cl_loss_pos(self, mask_positives, num_positives, share_cat):
        sim_share = self.similarity_measure(share_cat.unsqueeze(1), share_cat.unsqueeze(0)) / self.args.temperature
        temp_sim_share = torch.exp(sim_share)
        score_positives = temp_sim_share[mask_positives].sum() / num_positives
        return score_positives

    # 计算负样本得分
    def Cl_loss_neg(self, mask_dict, num_dict, data_dict, batch_size):

        # 4.每个样本的独有数据与所有视图的所有共享数据
        # sim_unique_share包括了独有数据与独有数据和共享数据与共享数据的相似度值
        sim_unique_share = self.similarity_measure(data_dict["unique_share_cat"].unsqueeze(1),
                                                   data_dict["unique_share_cat"].unsqueeze(0)) / self.args.temperature
        temp_sim_unique_share = torch.exp(sim_unique_share)
        score_unique_to_share = temp_sim_unique_share[mask_dict["mask_share_unique_noise"]].sum() / num_dict[
            "num_share_unique_noise"]

        # 1.每个样本的共享与不同视图的其他样本共享数据
        temp_sim_share = temp_sim_unique_share[self.args.num_view * batch_size:, self.args.num_view * batch_size:]
        score_share_to_other_share = temp_sim_share[mask_dict["mask_share_to_other_share"]].sum() / num_dict["num_share_to_other_share"]

        # 2.每个共享数据与所有噪音数据
        sim_share_noise = self.similarity_measure(data_dict["share_noise_cat"].unsqueeze(1),
                                                  data_dict["share_noise_cat"].unsqueeze(0)) / self.args.temperature
        temp_sim_share_noise = torch.exp(sim_share_noise)
        score_share_to_noise = temp_sim_share_noise[mask_dict["mask_share_unique_noise"]].sum() / num_dict["num_share_unique_noise"]

        # 3.每个独有数据与所有噪音数据
        sim_unique_noise = self.similarity_measure(data_dict["unique_noise_cat"].unsqueeze(1),
                                                   data_dict["unique_noise_cat"].unsqueeze(0)) / self.args.temperature
        temp_sim_unique_noise = torch.exp(sim_unique_noise)
        score_unique_to_noise = temp_sim_unique_noise[mask_dict["mask_share_unique_noise"]].sum() / num_dict[
            "num_share_unique_noise"]

        # 5.每个样本的独有数据与其他视图的所有独有数据
        temp_sim_unique = temp_sim_unique_share[0:self.args.num_view * batch_size, 0:self.args.num_view * batch_size]
        score_unique_to_unique = temp_sim_unique[mask_dict["mark_unique_to_unique"]].sum() / num_dict[
            "num_unique_to_unique"]

        neg = score_share_to_other_share + score_share_to_noise + score_unique_to_noise + score_unique_to_share + score_unique_to_unique
        return neg

    # 计算对比损失
    def ContrastiveLoss(self, share_list, noise_list, unique_list, batch_size):
        if self.args.lambda_cl == 0:
            return 0

        data_dict = dict()
        data_dict["share_cat"] = torch.cat(share_list, dim=0)  # 所有视图共享数据的拼接
        data_dict["noise_cat"] = torch.cat(noise_list, dim=0)  # 所有视图噪音数据的拼接
        data_dict["share_noise_cat"] = torch.cat((data_dict["share_cat"], data_dict["noise_cat"]), dim=0)  # 所有视图共享和噪音数据的拼接
        data_dict["unique_cat"] = torch.cat(unique_list, dim=0)  # 所有视图独有数据的拼接
        data_dict["unique_noise_cat"] = torch.cat((data_dict["unique_cat"], data_dict["noise_cat"]), dim=0)  # 所有视图独有和噪音数据的拼接
        data_dict["unique_share_cat"] = torch.cat((data_dict["unique_cat"], data_dict["share_cat"]), dim=0)  # 所有视图独有和共享数据的拼接

        if batch_size != self.args.batch_size:
            mask_dict = dict()
            num_dict = dict()
            mask_dict["mask_positives"], mask_dict["mask_share_to_other_share"], mask_dict["mask_share_unique_noise"], \
            mask_dict["mark_unique_to_unique"], num_dict["num_share_to_other_share"], num_dict["num_positives"], \
            num_dict["num_share_unique_noise"], num_dict["num_unique_to_unique"] = self.mark_matrix(self.args.view_num, batch_size)
            # 计算正得分
            pos = self.Cl_loss_pos(mask_dict["mask_positives"], num_dict["num_positives"], data_dict["share_cat"])

            # 计算负得分
            neg = self.Cl_loss_neg(mask_dict, num_dict, data_dict)

            # 计算对比损失
            cl_loss = -torch.log(pos / (pos + neg))
            return cl_loss * self.args.lambda_cl
        else:
            # 计算正得分
            pos = self.Cl_loss_pos(self.mask_dict["mask_positives"], self.num_dict["num_positives"], data_dict["share_cat"])

            # 计算负得分
            neg = self.Cl_loss_neg(self.mask_dict, self.num_dict, data_dict, batch_size)

            # 计算对比损失
            cl_loss = -torch.log(pos / (pos + neg))
            return cl_loss * self.args.lambda_cl

    # 计算重构损失
    def Reconstruction_loss(self, dec_list, x, num):
        rec_loss = 0.0
        for i in range(self.args.num_view):
            rec_loss = rec_loss + torch.sum(torch.pow(dec_list[i] - x[i], 2))
        rec_loss = rec_loss / (self.args.num_view * num)
        return rec_loss * self.args.lambda_rec

    # 计算VAE的KL散度
    def Vae_kl_loss(self, mu, logvar, num):
        vae_kl_loss = 0.0
        for i in range(self.args.num_view):
            vae_kl_loss = vae_kl_loss + 0.5 * torch.sum(torch.exp(logvar[i]) + torch.pow(mu[i], 2) - 1. - logvar[i])
        # print(vae_kl_loss)
        vae_kl_loss = vae_kl_loss / (self.args.num_view * num)
        return vae_kl_loss * self.args.lambda_vae_kl

    # 计算KL散度损失
    def Kl_loss(self, q, p):
        kl_loss = F.kl_div(q.log(), p.to(self.args.device), reduction='batchmean')
        return kl_loss * self.args.lambda_kl
