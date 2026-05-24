'''
        许多线性分类模型只适用于二分类问题，不能轻易推广到多类别问题（除了Logistic回归）。
    将二分类算法推广到多分类算法的一种常见方法是 “一对其余”（即one VS rest）方法。
    
        在 “一对其余” 方法中，对每个类别都学习一个二分类模型。将这个类别与所有其他类别尽量
    分开，这样就生成了与类别个数一样多的二分类模型。在测试点上运行所有二分类器来进行预测。在
    对应类别上分数最高的分类器 “胜出” ，将这个类别标签返回作为预测结果。
    
        每个类别都对应一个二分类器，这样每个类别也都有一个系数（W）向量和一个截距（b）。

        多分类 Logistic 回归背后的数学与 “一对其余” 方法稍有不同，但它也是对类别都有一个系数
    向量和一个截距，也使用了相同的预测方法。
'''


# 二维数据集 make_blobs 中每个类别的数据都是从一个高斯分布采样得出的。
from sklearn.datasets import make_blobs
import mglearn
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import numpy as np

X, y = make_blobs(random_state=42) # 设置随机种子，以便复现
# 查看数据集
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend(['Class 0', 'Class 1', 'Class 2'])
plt.show()

# 在该数据集上训练一个 LinearSVC 分类器
linear_svm = LinearSVC().fit(X, y)
# 查看系数形状
print('Coefficient shape：',linear_svm.coef_.shape)
# 查看截距形状
print('Intercept shape：', linear_svm.intercept_.shape)

# 将3个二类分类器给出的直线可视化。
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend(['Class 0', 
            'Class 1', 
            'Class 2', 
            'Line Class 0', 
            'Line Class 1',
            'Line Class 2'],
           loc = (1.01, 0.3))
plt.show()
'''
    多分类线性模型原理在该数据集上的运用：
    可以看到，训练集中所有属于类别 0 的点都在与类别 0 对应的直线上方，
    这说明它们位于这个二类分类器属于 ‘类别 0 ’ 的那一侧。
    属于类别 0 的点位于与类别 2 对应的直线上方，这说明它们被类别 2 的
    二类分类器划为‘其余’。属于类别 0 的点位于与类别 1 对应的直线左侧，
    说明类别 1 的二类分类器将它们化为‘其余’。
    
    图像中间的三角形区域属于哪一个类别呢？这里的点应该划归到哪一个类别呢？
    答：3个二类分类器都将这一区域内的点划为‘其余’。
        三角形区域内的点将会被划归到分类方程结果最大的那个类别，即最接近
        的那条线对应的类别。
'''
# 二维空间中所有区域的预测结果
mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=0.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.legend(['Class 0', 'Class 1', 'Class 2',
            'Line class 0', 'Line class 1', 'Line class 2',],
           loc = (1.01, 0.3))
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

'''
    线性模型的主要参数是正则化参数，在回归模型中叫做 alpha ，在 LinearSVC 和 Logistic Regression 
中叫做 C 。alpha的值较大或 C 值较小， 说明模型比较简单。但是对于回归模型而言，调节这些参数非常重要。
通常在对数尺度上对 C 和 alpha 进行搜索。
    还需要确定的是用 L1 正则化还是用 L2正则化。如果假定只有几个特征是真正重要的，那么应该使用 L1 正
则化，否则应该使用 L2 正则化。'''

'''
    线性模型的训练速度非常快，预测速度也很快。线性模型可以推广到非常大的数据集
    线性模型的另一个优点是便于理解。
'''