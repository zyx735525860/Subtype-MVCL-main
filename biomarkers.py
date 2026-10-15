import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# # STAD LUAD UCEC BLCA BRCA SKCM KIRC GBM PAAD UVM
data_floder = 'STAD'

labels_file = f'./result/cluster/{data_floder}.cluster'
file = f'./datasets/{data_floder}/rna.fea'
# feature_name = []
# 读取和合并组学数据
df = pd.read_csv(file, sep=',', header=0, index_col=0)
gene_id_column = df.index
df = df.T  # 转置 DataFrame
df = pd.DataFrame(df.values.astype(float))  # 转换为 DataFrame 对象
df.columns =gene_id_column

# 读取标签文件
labels = pd.read_csv(labels_file, sep='\t')
labels = labels.loc[:, ['Labels']]
# print(omics_data)
# 确保标签文件和组学数据的索引对齐
data = df.join(labels)

# 分离特征和标签
X = data.drop(columns=['Labels'])  # 特征
y = data['Labels']  # 标签

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(random_state=42)

# 使用网格搜索进行参数调优
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print('Best Parameters:', grid_search.best_params_)
best_rf = grid_search.best_estimator_

# 重新训练最佳模型
best_rf.fit(X_train, y_train)

# 使用交叉验证评估调优后的模型性能
scores = cross_val_score(best_rf, X_scaled, y, cv=5)
print('Cross-Validation Accuracy Scores:', scores)
print('Mean Cross-Validation Accuracy:', scores.mean())

# 获取基尼重要性分数
feature_importances = best_rf.feature_importances_

# 获取特征名称
feature_names = X.columns

importance_df = pd.DataFrame({'特征': feature_names, '重要性': feature_importances})
importance_df_sorted = importance_df.sort_values(by='重要性', ascending=False).reset_index(drop=True)

# 打印排序后的特征重要性分数
print(importance_df_sorted)





