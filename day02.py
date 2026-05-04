import numpy as np
from sklearn import datasets
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 加载手写数字数据集
digits = datasets.load_digits()
# 创建特征矩阵
features = digits.data
# 创建目标向量
target = digits.target
# 查看第一个样本值
print(features[0])

# 创建仿真数据集
"""如果需要用一个仿真数据集来做线性回归，可以用make_regression"""
# 生成特征矩阵、目标向量以及模型的系数
features, target, coefficients = make_regression(n_samples= 100,
                                                 n_features=3,
                                                 n_informative=3,
                                                 n_targets=1,
                                                 noise=0,
                                                 coef=True,
                                                 random_state=1)
# 查看特征矩阵和目标向量
print('Feature Matrix\n',features[:3])
print('Target Vector\n', target[:3])

"""如果需要创建一个仿真数据集来做分类，可以使用make_classfication"""
# 用分类特征创建特征矩阵
X = np.array([[0,2.1,1.45],
              [1,1.18,1.33],
              [1,-0.21,-1.19]])
print('X = \n',X)
# 创建带缺失值的特征矩阵
X_with_nan = np.array([[np.nan,0.87,1.31],
                           [np.nan,-0.67,-0.22]])
# 训练KNN分类器
clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(X[:,1:],X[:,0])
print(trained_model)
# 预测缺失集的分类
imputed_values = trained_model.predict(X_with_nan[:,1:])
# 将所预测的分类和它们的其他特征连接起来
X_with_imputed = np.hstack((imputed_values.reshape(-1,1), X_with_nan[:,1:]))
# 连接两个特征矩阵
print(np.vstack((X_with_imputed, X)))
