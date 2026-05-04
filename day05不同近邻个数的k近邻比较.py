from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# neighbors=1的情况
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()
# neighbors=3的情况
mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()

# 调用make_forge生成一个二维特征的二类分类数据集
x, y = mglearn.datasets.make_forge()
# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
# 导入类并将其实例化
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
# 对测试数据进行预测
print('Test set prediction:{}'.format(clf.predict(X_test)))
# 用score方法评估模型泛化能力好坏
print('Test set accuracy:{:.2f}'.format(clf.score(X_test, y_test)))

'''
    对于二维数据集，我们还可以在xy平面上画出所有可能的测试点的预测结果。
    根据平面中每个点所属的类别对平面进行着色，这样可以查看决策边界。即
    算法对类别0和类别1进行分界。
'''
fig, axes = plt.subplot(1, 3, figsize = (10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    # fit方法返回对象本身，所有可以将实例化和拟合放在一行代码中
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X=x, y=y)
    plt
    mglearn.plots.plot_2d_separator(clf, x, fill=True, eps=0.5, ax=ax, alpha=0.4)
    mglearn.discrete_scatter(x[:, 0], x[:, 1], y, ax=ax)
    plt.show()
    ax.set_title('{}neighbors(s)'.format(n_neighbors))
    ax.set_xlable('feature 0')
    ax.set_ylable('feature 1')
    axes[0].legend(loc=3)
