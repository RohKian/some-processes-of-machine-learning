"""
    假设有一名植物爱好者对她发现的鸢尾花的品种很感兴趣。她收集了每朵鸢尾花的一些测量数据，
    包括花瓣的长度和宽度，以及花萼的长度和宽度，所有测量结果的单位都是厘米。
    她还有一些鸢尾花的测量数据，这些花之前已经被植物学专家鉴定为属于setoa、versicolor、或
    virginica三个品种之一。
    
    假设这位植物爱好学者在野外只会遇到steoa、versicoloe、virginica这三种鸢尾花。
    我们目标是构建一个机器学习模型，可以从这些已知品种的鸢尾花测量数据中进行学习，从而能够
    预测新鸢尾花的品种。
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import plotting
import matplotlib.pyplot as plt
import mglearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris_dataset = load_iris()
# 查看数据集
print('key of iris_dataset:\n{}'.format(iris_dataset.keys()))
"""
   iris_dataset keys: data、target、frame、target_names、
                      DESCR、feature_names、filename、data_module
"""
# 查看数据集iris_dataset说明
print('iris_dataset description: ', iris_dataset['DESCR'])
# 查看花的样本数量
print('iris_dataset samples: {}'.format(iris_dataset['data'].shape))

'''描述性统计'''
# 查看前5行的数据
print('first five rows of data: \n{}'.format(iris_dataset['data'][:5]))
# 查看target, 其中0代表setoa，1代表versicolor，2代表virginica
print('target sample有{}个，target数据集为：\n{}'.format(iris_dataset['target'].shape,iris_dataset['target']))
'''
    衡量模型是否成功：训练数据与测试数据。
    想要利用该数据集构建一个机器学习模型，用于预测新测量的鸢尾花的品种。但在将模型应用于新的测量数据之前，
    我们需要知道模型是否有效。
    
    但是我们不能将用于构建模型的数据用于评估模型。
    因为模型会一直记住整个训练集，所以对于训练集中的任何数据点总会预测正确的标签。这种情况下该模型不具备泛化能力。
    一般是用新数据来评估模型的性能。新数据是指模型之前没有见过的数据，通常做法是将收集好的带标签的数据（此例中是150朵花的测量数据）
    分成两部分，一部分作为训练集，用来构建机器学习模型，其余的作为测试集，用来评估模型性能。
'''
# 利用python的第三方库的model_selection的train_test_split方法打乱训练集并进行拆分
# 这个模型将75%的行数据及对应标签作为训练集，剩下25%的数据作为测试集。（比例可调）
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0) #random_state=0，设置随机种子
# 查看data训练集样本容量和target训练集样本容量
print('X_train shape:{}\ny_train:{}'.format(X_train.shape, y_train.shape))
# 查看data测试集样本容量和target测试集样本容量
print('X_test shape:{}\ny_test:{}'.format(X_test.shape, y_test.shape))
'''
    观察数据
            在构建机器学习模型之前，通常最好检查一下数据，检查数据是否发现异常值和特殊值。
            检查数据的最佳方法之一是将其可视化。
            一种可视化方法是绘制散点图。（缺点是难以对多于3个特征的数据集作图）
        数据散点图将一个特征作为X轴，另一个特征作为y轴，将每一个数据点绘制为图上的一个点。
            另一个可视化方法是绘制散点图矩阵（可以解决对多于3个特征的数据集不能绘制散点图的问题）
'''
# 利用X_train中的数据创建DataFrame
# 利用iris_dataset.feature_name中的字符串对数据列进行标记
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 利用dataframe创建散点图矩阵，按y_train着色
grr = pd.plotting.scatter_matrix(frame=iris_dataframe, 
                                 c=y_train, # 与cmap配合生成颜色映射
                                 figsize=(15, 15), # 画布大小
                                 marker='o', # 点的形状
                                 hist_kwds={'bins':20}, # 传给对角线直方图的关键参数，数值越大，分布越细
                                 s=60, # 散点的大小，防止散点太密看不清
                                 alpha=0.8, # 透明度
                                 cmap = mglearn.cm3
                                 )
plt.show()

'''
    构建k近邻算法模型
    使用k近邻分类器。构建此模型只需保存训练集即可。要对一个新的数据点作出预测，算法会在训练集中寻找与这个新数据点距离最近（欧氏距离）
    的数据点，然后将找到的数据点的标签赋值给这个新数据点。    
'''
'''
    Knn对象对算法进行了封装，既包括训练数据构建模型的算法，也包括对新数据点进行预测的算法。
    但是对于KNeighborsClassifier（k近邻分类器）来说，只保存了训练集构建模型的算法。
'''
Knn = KNeighborsClassifier(n_neighbors=1) # 只参考距离最近的1个点
'''
    想要基于训练集来构建模型，需要调用knn对象的fit方法，输入参数为X_train和y_train，
    前者包含训练数据，后者包含相应德训练标签。
'''
Knn.fit(X_train, y_train) # fit方法只是把数据储存起来
# 进行预测测试
X_new = np.array([[5, 2.9, 1, 0.2]]) # 将新样本转化为数组
print('X_new shape:{}'.format(X_new.shape)) # 查看X_new形状
# 调用Knn对象德predict方法来预测
prediction = Knn.predict(X_new)
print('Prediction:{}'.format(prediction))
print('Prediction target name:{}'.format(iris_dataset['target_names'][prediction])) # Knn模型显示新样本属于类别0，即setosa

'''
    模型评估
    对于模型预测的结果是否值得相信的问题，我们可以通过对测试数据中的每朵鸢尾花进行预测，并将预测结果与标签（已知的品种）进行对比。
    通过计算精度来衡量模型的优劣。精度就是品种预测正确的花所占的比例。
'''
y_pred = Knn.predict(X_test)
print('Test set prediction:\n{}'.format(y_pred))
print('Test set score:{:.2f}'.format(np.mean(y_pred == y_test))) # 本质是sum(y_pred == y_test) / len(y_test)

