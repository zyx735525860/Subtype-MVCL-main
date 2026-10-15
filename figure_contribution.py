import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# 数据
cancer_types = ['STAD', 'LUAD', 'UCEC', 'BLCA', 'BRCA', 'SKCM', 'KIRC', 'GBM', 'PAAD', 'UVM']
meth = [0.521, 0.157, 0.251, 0.155, 0.167, 0.377, 0.213, 0.164, 0.116, 0.13]
mRNA = [0.354, 0.596, 0.314, 0.52, 0.549, 0.442, 0.432, 0.174, 0.756, 0.183]
CNV = [0.085, 0.21, 0.397, 0.282, 0.255, 0.143, 0.321, 0.662, 0.082, 0.669]
miRNA = [0.04, 0.037, 0.038, 0.043, 0.029, 0.038, 0.034, 0, 0.046, 0.018]

bar_width = 0.2
index = np.arange(len(cancer_types))

# 绘制条形图
fig, ax = plt.subplots(figsize=(12, 6))
bar1 = ax.bar(index, mRNA, bar_width, label='mRNA')
bar2 = ax.bar(index + bar_width, miRNA, bar_width, label='miRNA')
bar3 = ax.bar(index + bar_width*2, CNV, bar_width, label='CNV')
bar4 = ax.bar(index + bar_width*3, meth, bar_width, label='meth')

# 设置字体属性
font_properties_label = {'weight': 'bold', 'size': 14}
font_properties_xticks = {'size': 12}

# 添加标签和标题
ax.set_xlabel('Cancer Type', fontdict=font_properties_label)
ax.set_ylabel('Partial dependence', fontdict=font_properties_label)
ax.set_xticks(index + bar_width * 1.5)
ax.set_xticklabels(cancer_types, fontdict=font_properties_xticks)
ax.legend()

# 显示图表
plt.show()

