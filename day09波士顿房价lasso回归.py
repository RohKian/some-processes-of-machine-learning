'''
    除了 Ridge Regression （岭回归）之外，还有一种正则化的线性回归是 lasso 。
    与岭回归相同，使用lasso也是约束系数使其接近于 0 ，但用到的方法不同，叫做 L1
    正则化。L1正则化的结果是，使用lasso时某些系数刚好为 0，这样模型更容易解释，也
    可以呈现模型最重要的特征。
'''

from sklearn.linear_model import Lasso
from mglearn.datasets import load_extended_boston
from sklearn.model_selection import train_test_split
import numpy as np

x, y = load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
# lasso建模
lasso = Lasso().fit(X_train, y_train)
# 查看lasso模型在训练集和测试集上的分数
print('Training set score:{:.2f}'.format(lasso.score(X_train, y_train)))
print('Test set score:{:.2f}'.format(lasso.score(X_test, y_test)))
# 系数不为 0 的特征个数
print('Number of feature used:{}'.format(np.sum(lasso.coef_ != 0)))

'''
    Lasso 在训练集与测试集上的表现都很差，表明 Lasso 存在欠拟合。系数不为0
    的特征的个数只有4个。
    为了降低欠拟合，可以尝试减小 alpha 。同时还需要增加 max_iter （即运行迭代的最大次数）的值
'''
# 增大 max_iter 的值
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
# 查看 alpha 和 max_iter 改变后 lasso 在训练集和测试集的分数
print('\n lasso model alpha = 0.01，max_iter = 100000:')
print('Training set lasso001 score:{:.2f}'.format(lasso001.score(X_train, y_train)))
print('Test set lasso001 score:{:.2f}'.format(lasso001.score(X_test, y_test)))
print('Number of feature used:{}'.format(np.sum(lasso001.coef_ != 0)))

'''注意，如果把 alpha 值设得太小，那么就会消除正则化的效果，并出现过拟合。'''
'''
    在实践中，在岭回归和 lasso回归中一般首选岭回归。但如果特征很多，且个人认为只有其中几个
    是重要的，那么选择 lasso 回归可能更好。同样，如果想要一个容易理解的模型， lasso 模型
    可以给出更容易理解的模型。
'''
