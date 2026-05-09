import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 特征矩阵和目标值
X, y = mglearn.datasets.load_extended_boston()
# 数据集分类
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 构建线性回归模型
lr = LinearRegression().fit(X_train, y_train)
# 查看线性回归模型在训练集和测试集的分数
print('Training set score:{:.2f}'.format(lr.score(X_train, y_train)))
print('Test set score:{:.2f}'.format(lr.score(X_test, y_test)))

##### 总结 #####
'''可以看到，线性回归模型在训练集上的分数是0.95，而在测试集上的分数是0.61，
训练集和测试集分数差距过大，说明模型存在过拟合问题'''