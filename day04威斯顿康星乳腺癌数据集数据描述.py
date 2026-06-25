"""
    这是包含在scikit-learn中的一个威斯康星顿州乳腺癌数据集（简称caner），
    里面记录了乳腺癌肿瘤的临床测量数据。
    其中每个肿瘤都被标记为'良性'（begin，表示为无害肿瘤）或'恶性'（maligant，表示为癌性肿瘤），
    其任务是基于人体组织的测量数据来学习预测肿瘤是否为恶性。
"""

from sklearn.datasets import load_breast_cancer
import numpy as np

cancer = load_breast_cancer()
# 查看该威斯康星顿州乳腺癌数据集的关键字段
print('cancer dataset keys:\n{}'.format(cancer.keys()))
'''威斯康星顿州乳腺癌的关键字段有data，target，frame，target_names，DESCR，feature_names，filename，data_module'''
# 查看该数据集的大小
print('shape of cancer data:{}'.format(cancer['data'].shape)) # 该数据集包含569个数据点，每个数据点包含30个特征
# 查看该数据集中被标记为恶性的肿瘤个数和良性的肿瘤个数
print('sample counts per class:\n{}'.format({
    str(n) : str(v) for n, v in zip(cancer['target_names'], np.bincount(cancer['target']))
}))
# 查看该数据集德特征
print('Feature names:\n{}'.format(cancer['feature_names']))
# 查看该数据集德说明
print('cancer dataset description:\n{}'.format(cancer['DESCR']))
