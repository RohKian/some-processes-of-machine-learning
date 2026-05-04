# day01
# 创建向量
import numpy as np
vector_row = np.array([1,2,3])
vector_column = np.array([[1],
                          [2],
                          [3]])
print(f'vector_row = {vector_row}')
print(f'vector_column = \n{vector_column}')
# 创建一个矩阵
matrix = np.array([[1,2],
                   [1,3],
                   [1,2]])
print(f'matrix = \n{matrix}')
# 选择元素
print(f'第2行第2列元素为{matrix[1,1]}')
# 展示一个矩阵的属性
matrix = np.array([[1,2,3,4],
                   [5,6,77,8],
                   [9,10,11,12]])
print(f'矩阵matrix = \n{matrix}')
# 查看行数和列数
print(f"matrix行列数为{matrix.shape}")
# 查看元素的数量
print(f'matrix的元素数量为{matrix.size}')
# 查看矩阵的维数
print(f'matrix的维数为{matrix.ndim}维')
# 对多个元素同时应用某个操作
matrix = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]])
print(f'matrix = \n{matrix}')
# 创建向量化函数
vectorized_add_100 = np.vectorize(lambda i: i + 100)
# 对矩阵的所有数应用这个函数
matrix_added_100 = vectorized_add_100(matrix)
print(f'matrix_added_100 = \n{matrix_added_100}')
print(matrix + 100)
# 返回矩阵最大的元素
print(f'matrix中最大的元素为{np.max(matrix)}')
# 返回矩阵最小的元素
print(f'matrix中最小的元素为{np.min(matrix)}')
# 返回平均值
print(f'matrix中所有元素的平均值为{np.mean(matrix)}')
# 返回方差
print(f'matrix方差为{np.var(matrix):.2f}')
# 返回标准差
print(f'matrix标准差为{np.std(matrix):.2f}')
# 求每一列的平均值
print(f'matrix每一列的平均值为{np.mean(matrix, axis=0)}')
# 矩阵变形
matrix = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9],
                   [10,11,12]])
# 将matrix变为2 × 6的矩阵
print(matrix.reshape(2,6))
print(matrix)
# 转置矩阵matrix
print(f'转置后的matrix为{matrix.T}')
matrix = np.array([[1,1,1],
                   [1,1,10],
                   [1,1,15]])
# 计算matrix的秩
print(f'矩阵matrix的秩为{np.linalg.matrix_rank(matrix)}')
# 计算矩阵的行列式
print(f'matrix的行列式为{np.linalg.det(matrix)}')
matrix = np.array([[1,2,3],
                   [2,4,6],
                   [3,8,9]])
# 返回对角线元素
print(matrix.diagonal())
# 返回对角线向上偏移量为1的对角线元素
print(matrix.diagonal(offset=1))
# 返回对角线向下偏移量为1的对角线元素
print(matrix.diagonal(offset=-1))
# 计算矩阵的迹
print(matrix.trace())
matrix = np.array([[1,-1,3],
                   [1,1,6],
                   [3,8,9]])
print(f'matrix = \n{matrix}')
# 计算特征值和特征向量
eigen_values, eigen_vectors = np.linalg.eig(matrix)
print(f'矩阵matrix的特征值为：\n{np.round(eigen_values, 2)}\n特征向量为：\n{np.round(eigen_vectors, 2)}')

vector_a = [1,2,3]
vector_b = [4,5,6]
print(f'向量vector_a = \n{vector_a}')
print(f'向量vector_b = \n{vector_b}')
# 计算两个向量的点积（两矩阵相乘）
print(f'向量vector_a和向量vector_b的点积为{np.dot(vector_a,vector_b)}')

matrix = np.array([[1,4],
                   [2,5]])
# 计算矩阵的逆
print(f'矩阵matrix的逆为\n{np.round(np.linalg.inv(matrix), 2)}')

# 生成随机数
np.random.seed(0) # 设置随机数种子，确保实验结果可复现
# 生成3个0到1之间的随机浮点数
print(np.random.random(3))
# 生成3个1到10之间的随机整数
print(np.random.randint(1,11,3))
# 从平均值是0且标准差是1的正态分布中抽取3个数
print(np.random.normal(0,1,3))
# 从大于或等于1并且小于2的范围中抽取3个数
print(np.random.uniform(1,2,3))
