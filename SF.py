import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
# # STAD LUAD UCEC BLCA BRCA SKCM KIRC GBM PAAD UVM
data_floder = 'PAAD'

# 文件路径设置
omics_files = [
    f'./datasets/{data_floder}/meth.fea',
    f'./datasets/{data_floder}/rna.fea',
    f'./datasets/{data_floder}/CN.fea',
    f'./datasets/{data_floder}/miRNA.fea'

]
labels_file = f'./result/cluster/{data_floder}.cluster'

# 读取和合并组学数据
data_list = []
data_dim = []
for file in omics_files:
    df = pd.read_csv(file, sep=',', header=0, index_col=0)
    df = df.T  # 转置 DataFrame
    df = pd.DataFrame(df.values.astype(float))  # 转换为 DataFrame 对象
    data_dim.append(df.shape[1])
    data_list.append(df)

# 合并所有组学数据
# print(data_dim)
omics_data = pd.concat(data_list, axis=1)
unique_column_names = [f'v{i}' for i in range(omics_data.shape[1])]
omics_data.columns = unique_column_names
# print(omics_data)

# 读取标签文件
labels = pd.read_csv(labels_file, sep='\t')
labels = labels.loc[:, ['Labels']]
# print(omics_data)
# 确保标签文件和组学数据的索引对齐
data = omics_data.join(labels)
# print(data)

# 分离特征和标签
X = data.drop(columns=['Labels'])  # 特征
y = data['Labels']  # 标签

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 初始化并训练随机森林模型
rf = RandomForestClassifier(max_depth=6, random_state=42)
rf.fit(X_train, y_train)

# 获取基尼重要性分数
feature_importances = rf.feature_importances_

# 获取特征名称
feature_names = X.columns
# print(feature_names)

importance_df = pd.DataFrame({'特征': feature_names, '重要性': feature_importances})
# print(importance_df)

# 按组学数据类型分组计算重要性
omics_types = {
    'meth': feature_names[:data_dim[0]],
    'mRNA': feature_names[data_dim[0]:data_dim[0] + data_dim[1]],
    'CNV': feature_names[data_dim[0] + data_dim[1]:data_dim[0] + data_dim[1] + data_dim[2]],
    'miRNA': feature_names[data_dim[0] + data_dim[1] + data_dim[2]:]
}
# print(omics_types)
# print(omics_types['miRNA'])
# 计算每种组学数据的总重要性
importance_by_omics = {}
for omics_type, features in omics_types.items():
    # print(features)
    # print(importance_df[importance_df['特征'].isin(features)]['重要性'].sum())
    total_importance = importance_df[importance_df['特征'].isin(features)]['重要性'].sum()
    importance_by_omics[omics_type] = total_importance

# 打印结果
print("每种组学数据对亚型识别结果的贡献：")
for omics_type, importance in importance_by_omics.items():
    print(f"{omics_type}: {importance}")


