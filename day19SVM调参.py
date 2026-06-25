# 用支持向量机对一个二分类数据集 forge 进行训练
# 决策边界用黑色表示，支持向量是尺寸较大的点

from sklearn.svm import SVC
import matplotlib.pyplot as plt
import mglearn

X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
mglearn.plots.plot_2d_separator(svm, X, eps=0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# 画出支持向量
sv = svm.support_vectors_
# 支持向量的类别标签由 dual_coef_的正负号给出
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

'''
        在该例子中， SVM 给出了非常平滑且非线性（非直线）的边界。
    支持向量非常重要的两个参数：C 参数和 gamma参数。
        gamma参数：用于控制高斯核的宽度。它决定了点与点之间 “靠近” 是指多大的距离。
        C 参数：C参数是正则化参数，与线性模型中用到的类似。它限制了每个点的重要性（更准确地说，
                C 参数限制的是每个点的dual_coef_，即对偶问题中支持向量对应的系数。）
   '''
# 不同 C 和 gamma 的参数对应的决策边界和支持向量
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
axes[0, 0].legend(['class 0', 'class 1', 'sv class 0', 'sv class 1'],
                  ncol=4,
                  loc=(0.9, 1.2))
plt.show()
