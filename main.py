import argparse
import torch
import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息

from Model.Trainer.Models_trainer import model_trainer

if __name__ == "__main__":
    # 设置参数
    parser = argparse.ArgumentParser()
    # STAD LUAD UCEC BLCA BRCA SKCM KIRC GBM PAAD UVM
    parser.add_argument('--data_folder', type=str, default="KIRC")
    parser.add_argument('--pre_epoch', type=int, default=300)  # LUAD=300
    parser.add_argument('--eval_epoch', type=int, default=50)
    parser.add_argument('--num_epoch', type=int, default=1200)  # 论文 3.1.5 节：自监督微调 1200 轮
    parser.add_argument('--num_view', type=int, default=4)  # meth rna CN miRNA
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--alpha', type=float, default=1.)  # q分布的超参数
    parser.add_argument('--stop', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--update_interval', type=int, default=50)

    parser.add_argument('--hidden_dim', type=int, default=[1024, 512], help="MLP维度")
    parser.add_argument('--mu_logvar_dim', type=int, default=128)  # 128
    parser.add_argument('--noise_share_unique_dim', type=int, default=64)  # 64

    parser.add_argument('--dropout', type=float, default=0.2)  # 0.2
    parser.add_argument('--scheduler', type=str, default="CosineAnnealingLR")  # get_cosine_schedule_with_warmup # CosineAnnealingLR
    parser.add_argument('--act', type=str, default="leakyrelu") # leakyrelu
    parser.add_argument('--star_mutl_dim', type=int, default=2)
    parser.add_argument('--head_num', type=int, default=1)
    parser.add_argument('--lambda_rec', type=float, default=0.01)
    parser.add_argument('--lambda_kl', type=float, default=0.06)  # 0.06
    parser.add_argument('--lambda_cl', type=float, default=10.)  # 10
    parser.add_argument('--lambda_vae_kl', type=float, default=10.)  # 10

    parser.add_argument('--temperature', type=float, default=0.95)  # 对比损失的超参数
    parser.add_argument('--noise_intensities', type=float, default=0.05)  #
    args = parser.parse_args()
    cancer_dict = {'BRCA': 5, 'BLCA': 5, 'KIRC': 4,
                   'GBM': 4, 'LUAD': 3, 'PAAD': 2,
                   'SKCM': 4, 'STAD': 3, 'UCEC': 4, 'UVM': 4}
    args.cluster_number = cancer_dict[args.data_folder]
    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')

    # 建立模型
    model_trainer = model_trainer(args)

    # 模型训练
    model_trainer.train(need_pre_train=True)

    # 可视化
    # model_trainer.show()