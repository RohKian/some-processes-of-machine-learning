'''
    最常用的两种线性分类算法是 Logistic 回归和支持向量回归（即线性SVM），
    前者在 linear_model.LogisticRegression 中实现，后者在 svm.LinearSVC（SVC代表支持向量分类器）
    中实现。虽然 LogisticRegression 的名字中含有回归（regression），但它是一种分类算法，并不是回归
    算法。
'''
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import mglearn
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=0.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title('{}'.format(clf.__class__.__name__))
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')

axes[0].legend()
plt.show()

'''不同 C 值的线性 SVM 在 forge 的数据集上的决策边界'''
'''
    C 值很小，对应强正则化。强正则化的模型会选择一条相对水平的线。
    C 值稍大，模型更关注两个分类错误的样本，使决策边界的斜率变大。
    
    与回归的情况类似，用于分类的线性模型在低维空间中看起来可能非常受限，决策边界只能是直线或平面。
    同样，在高维空间中，用于分类的线性模型变得非常强大，当考虑更多特征时，避免过拟合非常重要！
'''