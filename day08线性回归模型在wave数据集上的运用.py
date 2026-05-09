import mglearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

mglearn.plots.plot_linear_regression_wave()
plt.show()

'''
    从wave[0]可以看出，斜率为0.393906，大概在0.4左右，截距为-0.031804，
    
    用于回归的线性模型可以表示为这样的回归模型：对单一特征的预测结果是一条直线，
    两个特征是一个平面，或在更高维度（即更多特征）时是一个超平面。
    
    线性回归，即使用最小二乘法寻找参数 W 和 b，使得对训练集的预测值与真实的回归目标值 y 之间的均方误差
    是预测值与真实值之差的平方和除以样本数。
'''
x, y = mglearn.datasets.make_wave(n_samples = 60)
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)

'''
    斜率参数（w，也叫作权重或系数）被保存在coef_属性中，
    而偏移或截距(b)被保存在intercept_属性中。
'''
# 查看线性回归系数
print('lr.coef_:{}'.format(lr.coef_))
print('lr.intercept_:{}'.format(lr.intercept_))
# 查看训练集和测试集性能
print('Training set score:{:.4f}\n Test set score:{:.4f}'.format(lr.score(X_train, y_train), lr.score(X_test, y_test)))
