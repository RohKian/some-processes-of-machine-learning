'''
    朴素贝叶斯分类器与线性模型分类器相似，但朴素贝叶斯的训练速度往往更快。
    这种高效付出的代价是朴素贝叶斯模型的泛化能力要比线性分类器稍差。
    
    朴素贝叶斯模型高效的原因在于：它通过单独查看每个特征来学习参数，并从每个特征中
    收集简单的类别统计数据。Scikit-learn 实现了三种朴素贝叶斯分类器：
    GaussianNB、BernoulliNB、MultinomialNB。
    其中 GaussianNB 可应用于任意连续数据，而 BernoulliNB 假定输入数据为二分类数据，
    MultinomialNB 假定输入数据为计数数据（即每个特征代表某个对象的整数计数，如一个单词
    在句子里出现的次数）。BernoulliNB 和 MultinomialNB 主要用于文本数据分类。 
'''

import numpy as np
# BernoulliNB 分类器计算每个类别中每个特征不为 0 的元素个数
X = np.array([[0, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])
counts = {}
# 计算每个特征中 1 的个数
for label in np.unique(y):
    counts[label] = X[y == label].sum(axis=0)
print('Feature counts：\n{}'.format(counts))
''' 
    MutinomialNB 计算每个类别中每个特征的平均值，而 GussianlNB 会保存每个类别中每个
    特征的平均值和标准差。

优点、缺点和参数
    MutinomialNB 和 BernoulliNB 都只有一个参数 alpha，用于控制模型复杂度。alpha的工作
    原理是：算法向数据中添加 alpha 这么多的虚拟数据点，这些点对所有特征都取正值，这样可以
    将统计数据‘平滑化’。alpha越大，平滑化越强，模型复杂度就越低。算法性能对alpha值的鲁棒性
    较好，也就是说，alpha 值对模型性能并不重要。但调整这个参数通常都会使精度略有提高。
'''
