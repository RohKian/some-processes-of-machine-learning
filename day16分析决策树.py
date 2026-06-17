# 利用 tree 模块的 export_graphviz 函数来将树可视化
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import mglearn

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer['data'],
    cancer['target'],
    stratify=cancer['target'],
    random_state=42
)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train,y_train)
export_graphviz(
    tree, 
    out_file='tree.dot', 
    class_names=['malignant','benign'],
    feature_names=cancer.feature_names,
    impurity=False,
    filled=True
)

with open('tree.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
'''
        树的可视化有助于深入理解算法是如何进行预测的，但是有一种观察树
    的方法可能有用，就是找出大部分数据的实际路径。
    
    一：直接观察法
        tree.png 中每个结点的 samples 都给出了该结点中的样本个数,values
    给出的是每个类别的样本个数，观察根结点 worst radius <= 16.795分支
    右侧的子结点 texture error <= 0.473，其中 value = [134, 8]，表示
    有 134个恶性样本，8个良性样本。观察根结点 worst radius <= 16.795分支
    左侧的子结点 worst concave points <= 0.136，表示有 25个恶性样本，259
    个良性样本。
        几乎所有的良性样本最终都进入左数第二个叶结点中，大部分其他叶结点都只包含很少
    的样本。
    
    二：树的特征重要性
        观察整个树，查看数据的实际路径很难。因此可以利用一些有用的属性来总结树的工作原理。
    其中最常用的是特征重要性。决策树的特征重要性为每个特征对树的决策的重要性进行排序。对
    于每个特征来说，它都是一个介于 0 和 1 之间的数字，其中 0 表示 “根本没用到”， 1 表示
    “完美预测目标值”。特征重要性的求和始终为 1。
'''
print('Feature importance：\n{}'.format(tree.feature_importances_))

# 可视化特征重要性
def plot_feature_importance_cancer(model):
    n_features = cancer['data'].shape[1]
    plt.barh(range(n_features),
             model.feature_importances_,
             align= 'center')
    plt.yticks(np.arange(n_features),
               cancer.feature_names)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.show()

plot_feature_importance_cancer(tree)
tree = mglearn.plots.plot_tree_not_monotone()
plt.show()

'''
    决策树有两个优点：
    1：得到的模型很容易可视化
    2：算法完全不受数据缩放的影响。由于每个特征被单独处理，而且数据的划分也不依赖于缩放，
       因此决策树算法不需要特征预处理，比如归一化或标准化。
       特征的尺度完全不一样时或者二元特征和连续特征同时存在时，决策树的效果很好。
    
    决策树的主要缺点在于：即使做了预剪枝，也经常会出现过拟合，泛化性能很差。
    通常用 随机森林 和 梯度回归树（梯度提升机）集成的方法来替代单棵决策树。
'''
