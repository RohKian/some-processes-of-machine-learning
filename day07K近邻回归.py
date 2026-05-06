'''
    K近邻算法还可以用于回归。利用单一邻居的预测结果就是最近邻的目标值。
'''
import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# 单个近邻回归
mglearn.plots.plot_knn_regression(n_neighbors=1)
plt.show()
# 多个近邻回归
mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()

x, y = mglearn.datasets.make_wave(n_samples = 40)
# 将wave数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
# 模型实例化，将邻居个数设置为3
reg = KNeighborsRegressor(n_neighbors=3)
# 利用训练数据和训练目标值来拟合模型
reg.fit(X_train, y_train)
# 对测试集进行预测
print('Test set prediction:\n{}'.format(reg.predict(X_test)))
# 查看精度，即R^2
print('Test set R^2:{:.2f}'.format(reg.score(X_test, y_test)))

# 分析KNeighborsRegressor
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# 创建1000个数据点，在-3和3之间分布均匀
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # 利用1个、3个或9个邻居分别进行预测:
    reg = KNeighborsRegressor(n_neighbors=n_neighbors,)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.set_title('{} neighbor(s)\n train score: {:.2f} test score:{:.2f}'.format(n_neighbors,reg.score(X_train, y_train), reg.score(X_test, y_test)))
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
    axes[0].legend(['Model predicyion', 'Training data/target', 'Test data/target'], loc='best')
plt.show() 
'''
    仅使用单一邻居，训练集中的每个点都对预测结果有显著影响，预测结果的图像经过所有数据点，
    导致预测结果非常不稳定。考虑更多的邻居后，预测结果变得平滑，但对训练数据的拟合不好。
'''
