'''
    通常来说，构造决策树直到所有的叶结点都是纯的叶结点，通常会导致模型非常复杂，并且对训练数据
    高度过拟合。纯叶结点的存在说明所构造的决策树在该训练集上的精度达到了100%。
    
    “决策树中的过拟合现象”
    如果在决策树中属于类别 0 的点中间有一块属于类别 1 的区域。另一方面，有一条属于类别 0 的区域，
    且出现了包围着属于类别为 0 的点的现象。这就是决策树中的过拟合现象。
    
    “决策树中防止过拟合现象的两种策略”
    1：及早停止树的生长，也叫预剪枝（pre-pruning）。
       预剪枝的限制条件可能包括限制树的最大深度、限制叶结点的最大数目，或者规定一个结点中数据点的
       最小数目来防止继续划分。
    2：后剪枝叶（post-pruning）
       先构造树，随后删除或折叠信息很少的结点。
'''

# 利用乳腺癌数据集来查看预剪枝的效果
# 利用默认设置构建决策树，默认将树全部展开（树不断分支，直到所有的叶结点都是纯的）
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test =train_test_split(
    cancer['data'],
    cancer['target'],
    stratify = cancer['target'],
    random_state=42,
)
# 方法一：预剪枝
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print('Way 1：')
print('Accuracy on training set：{:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on test set：{:.3f}'.format(tree.score(X_test, y_test)))

'''
    Accuracy on training set：1.000，
    Accuracy on test set:0.937
    
    因为叶结点都是纯的，树的深度很大，足以完美地记住训练数据的所有标签.如果不限制决策树的深度，
    那么它的深度和复杂度将会变得特别大。因此，未剪枝的树容易过拟合。对新数据的泛化性能不佳。
    将预剪枝应用到决策树上，这可以在完美拟合训练数据前组织树的展开。
    
    另一种选择是在达到一定深度后停止树的展开。
    例如设置 max_depth = 4，意味着只可以连续问4个问题。限制树的深度可以减少过拟合。
    这会降低训练集的精度，但可以提高测试集的精度（如何trade off，确实是一个问题。）
'''
# 方法二：主动设置问题深度，达到深度后自动停止
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print('\nWay 2：')
print('Accuracy on training set：{:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on test set：{:.3f}'.format(tree.score(X_test, y_test)))
