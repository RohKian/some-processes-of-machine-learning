# 利用 tree 模块的 export_graphviz 函数来将树可视化
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz

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