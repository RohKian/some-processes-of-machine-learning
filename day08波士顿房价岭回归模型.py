import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# 特征矩阵和目标值
X, y = mglearn.datasets.load_extended_boston()
# 数据集分类
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 构建线性回归模型
lr = LinearRegression().fit(X_train, y_train)
# 查看线性回归模型在训练集和测试集的分数
print('Linear regression result')
print('Training set score:{:.2f}'.format(lr.score(X_train, y_train)))
print('Test set score:{:.2f}'.format(lr.score(X_test, y_test)))

##### 总结 #####
'''
    可以看到，线性回归模型在训练集上的分数是0.95，而在测试集上的分数是0.61，
    训练集和测试集分数差距过大，说明模型存在过拟合问题。
    标准线性回归最常用的替代方法之一就是岭回归。
'''
'''
    在岭回归中，对系数的选择不仅要在训练数据上得到好的结果预测结果，而且还要拟合附加约束。
    系数应该尽量小，换句话说就是 w 的所有元素都应该接近于 0 。这意味着每个特征对于输出的
    影响尽可能小（即斜率很小）。这种约束是所谓正则化的一个例子。
    正则化是指对模型做显示约束，以免过拟合。
    岭回归用到的正则化称为 L2 正则化。
'''
# 对训练集进行岭回归建模
ridge = Ridge().fit(X_train, y_train)
# 查看模型在训练集和测试集上的分数
print('ridge regression result:')
print('Training set score:{:.2f}'.format(ridge.score(X_train, y_train)))
print('Test set score:{:.2f}'.format(ridge.score(X_test, y_test)))

'''
    Ridge Regression在训练集上的分数要低于Linear Regression，但在测试集上的
    分数更高。Ridge约束性更强，更不容易过拟合，但它复杂度更小，意味着在训练集上
    的性能更差，但泛化性能更好。
    
    简单性和复杂性能二者对于模型的重要程度可以通过用户设置alpha参数来指定。alpha的
    最佳设定取决于用到的具体数据集。
    增大alpha会使得系数更加趋向于 0 ，从而降低训练集性能，但可能会提高泛化。
    减小alpha可以让系数受到的限制更小。
'''