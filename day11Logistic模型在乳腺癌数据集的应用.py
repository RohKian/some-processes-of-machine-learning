from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()
# 其中stratify=cancer['target']是为了让测试集和训练集的类别分布一致
X_train, X_test, y_train, y_test = train_test_split(cancer['data'],
                                                    cancer['target'],
                                                    stratify=cancer['target'],
                                                    random_state=42)
logreg = LogisticRegression().fit(X_train, y_train)
print('LogisticRegression(c=1)')
print('Training set score:{:.3f}'.format(logreg.score(X_train, y_train)))
print('Test set score{:.3f}'.format(logreg.score(X_test, y_test)))

# 尝试增大 C 值来查看不同 C 值的 LogisticRegression 在训练集和测试集的拟合状况。
# C = 100
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print('LogisticRegression(c=100)')
print('Training set score:{:.3f}'.format(logreg.score(X_train, y_train)))
print('Test set score:{:.3f}'.format(logreg.score(X_test, y_test)))

# C = 0.01
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print('LogisticRegression(c=0.01)：')
print('Training set score:{:.3f}'.format(logreg001.score(X_test, y_test)))
print('Test set score:{:.3f}'.format(logreg001.score(X_test, y_test)))

