import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 创建数据框
data = {
    "Noise": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    "DIMIOR": [3.9, 3.6, 3.4, 3.4, 3.5, 3.2, 3.4, 3.3, 3.1, 3, 3.1],
    "DIMIOR-SA": [2.4, 1.9, 2, 1.7, 1.8, 1.7, 1.5, 1.3, 1.4, 1.2, 1],
    "Kmeans": [0.6, 1, 0.9, 0.5, 0.1, 0.2, 0.07, 0.08, 0.06, 0.04, 0.01],
}

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# Melt数据框到长格式
df_long_p = pd.melt(df, id_vars=["Noise"], value_vars=["DIMIOR", "DIMIOR-SA", "Kmeans"],
                    var_name="Method", value_name="P-value")

# 设置线型和点型
markers = {"DIMIOR": "o", "DIMIOR-SA": "^", "Kmeans": "s"}
colors = {"DIMIOR": "red", "DIMIOR-SA": "green", "Kmeans": "blue"}

# 创建绘图
fig, ax = plt.subplots()

# 绘制每一条线
for method in markers:
    subset = df_long_p[df_long_p["Method"] == method]
    ax.plot(subset["Noise"], subset["P-value"], marker=markers[method], color=colors[method], label=method)

# 设置标题和轴标签
ax.set_xlabel("Noise intensity")
ax.set_ylabel("-log10 of  logrank P-value")

ax.set_ylim(0, 4.7)

# 添加百分比形式的x轴标签
ax.set_xticks(df["Noise"])
ax.set_xticklabels([f"{x}%" for x in df["Noise"]])

# 添加图例
legend = ax.legend(loc='upper right', bbox_to_anchor=(1.01, 1), frameon=False)

# 为图表添加边框
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # 设置黑色边框

# 创建图例边框
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.5)

plt.tight_layout()
plt.show()

