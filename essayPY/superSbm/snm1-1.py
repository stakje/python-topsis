import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 读取 Excel 文件
def read_excel(file_path):
    df = pd.read_excel(file_path, index_col=0)
    return df.values

# SuperSBM 模型定义
class SuperSBM:
    def __init__(self, n, k):
        """
        初始化 Super SBM 模型
        :param n: 节点总数
        :param k: 社区数量
        """
        self.n = n
        self.k = k
        self.memberships = np.random.dirichlet(np.ones(k), size=n)  # 初始化社区分配
        self.p_in = np.random.rand(k)  # 初始化社区内部连接概率
        self.p_out = np.random.rand(k, k)  # 初始化社区间连接概率
        np.fill_diagonal(self.p_out, self.p_in)  # 确保社区内部概率设置正确

    def compute_likelihood(self, adj_matrix):
        """
        计算当前模型的对数似然
        :param adj_matrix: 邻接矩阵
        :return: 对数似然值
        """
        likelihood = 0.0
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    p_ij = np.dot(self.memberships[i], np.dot(self.p_out, self.memberships[j]))
                    if adj_matrix[i, j] == 1:
                        likelihood += np.log(p_ij)
                    else:
                        likelihood += np.log(1 - p_ij)
        return likelihood

    def e_step(self, adj_matrix):
        """
        E 步骤: 更新社区分配
        :param adj_matrix: 邻接矩阵
        """
        for i in range(self.n):
            for j in range(self.k):
                p_ij = np.dot(self.memberships[i], np.dot(self.p_out, self.memberships[j]))
                self.memberships[i][j] = np.exp(np.log(p_ij) - np.log(1 - p_ij))
            self.memberships[i] /= np.sum(self.memberships[i])  # 归一化

    def m_step(self, adj_matrix):
        """
        M 步骤: 更新社区内外连接概率
        :param adj_matrix: 邻接矩阵
        """
        self.p_in = np.zeros(self.k)
        self.p_out = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                num = 0
                den = 0
                for u in range(self.n):
                    for v in range(self.n):
                        if u != v:
                            num += adj_matrix[u, v] * self.memberships[u][i] * self.memberships[v][j]
                            den += self.memberships[u][i] * self.memberships[v][j]
                self.p_out[i][j] = num / den
        np.fill_diagonal(self.p_out, self.p_in)

    def fit(self, adj_matrix, max_iter=100):
        """
        拟合模型
        :param adj_matrix: 邻接矩阵
        :param max_iter: 最大迭代次数
        """
        for _ in range(max_iter):
            self.e_step(adj_matrix)
            self.m_step(adj_matrix)
            likelihood = self.compute_likelihood(adj_matrix)
            print(f"Log Likelihood: {likelihood}")

# 示例 Excel 文件路径
file_path = 'data.xlsx'
adj_matrix = read_excel(file_path)
n = adj_matrix.shape[0]
k = 3  # 假设有3个社区

# 初始化和拟合模型
model = SuperSBM(n, k)
model.fit(adj_matrix)

# 绘制邻接矩阵
plt.imshow(adj_matrix, cmap='Greys', interpolation='none')
plt.title("Adjacency Matrix from Excel")
plt.show()
