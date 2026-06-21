'''
    核支持向量机（kernelized support vector machine）
        核支持向量机（通常简称为 SVM）,是可以推广到更复杂模型的扩展，
    这些模型无法被输入空间的超平面定义。
    
    1.线性模型和非线性特征
        线性模型在低维空间中可能非常受限，因为线和平面的灵活性有限。有
    一种方法可以让线性模型更加灵活，就是添加更多的特征。
'''
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np
import mglearn

X, y = make_blobs(centers=4, random_state=8)
y = y % 2
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

# 二分类数据集，其类别并不是线性可分的
linear_svm = LinearSVC().fit(X, y)
mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

# 添加第二个特征的平方，作为一个新特征
X_new = np.hstack([X, X[:, 1:] ** 2])
figure = plt.figure()

# 3D可视化（新版 matplotlib 推荐写法）
ax = figure.add_subplot(111, projection='3d', elev=-152, azim=-26)

mask = y == 0
ax.scatter(X_new[mask, 0],
           X_new[mask, 1],
           X_new[mask, 2],
           c='b',
           marker='o',
           s=60)
ax.scatter(X_new[~mask, 0],
           X_new[~mask, 1],
           X_new[~mask, 2],
           c='r',
           marker='^',
           s=60)
ax.set_xlabel('feature0')
ax.set_ylabel('feature1')
ax.set_zlabel('feature1 ** 2')
plt.show()

'''
    核技巧
        向数据表示中添加非线性特征，可以让线性模型变得强大。但是，通常来说
    我们并不知道要添加哪些特征，而且添加许多特征（比如100维特征空间所有
    可能的交互项）的计算开销可能会很大。但是，有一种巧妙的数学技巧，让我
    们可以在更高维空间中学习分类器，而不用实际计算可能非常大的新的数据表示。
    这种技巧叫作核技巧（kernel trick）。
        它的原理是：直接计算扩展特征表示中数据点之间的距离（更准确地说是内积），
    而不用实际对扩展进行计算。
        对于支持向量机，将数据映射到更高维空间中有两种常用的方法：
        1：多项式核，在一定阶数内计算原始特征所有可能的多项式；
        2：径向基函数（radial basis function，RBF）核，也叫高斯核。高斯核对应
        无限维的特征空间。（高斯核：考虑所有阶数的所有可能的多项式，但阶数越高，
        特征的重要性越小）
        
    核支持向量机（SVM）的训练原理
        在训练过程中，SVM 学习每个训练数据点对于两个类别之间决策边界的重要性。
    通常只有一部分训练数据点对于定义决策边界来说很重要：位于类别之间边界上的那些点
    叫作支持向量，支持向量机由此得名。
        想要对新样本点进行预测，需要测量它与每个支持向量之间的距离。分类决策是基于它与
    支持向量之间的距离以及在训练过程中学到的支持向量重要性（保存在 svc 的dual_coef_属性中）
    来做出的。数据点之间的距离由高斯核给出。
'''
