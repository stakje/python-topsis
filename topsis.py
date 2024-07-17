import numpy as np

# 示例数据：m个城市，n个指标
C = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# 数据标准化处理
def standardize_data(C):
    # 正向指标标准化
    min_C = np.min(C, axis=0)
    max_C = np.max(C, axis=0)
    Z = (C - min_C) / (max_C - min_C + 0.00001)
    return Z

Z = standardize_data(C)

# 计算信息熵和冗余度
def calculate_entropy(Z):
    m, n = Z.shape
    P = Z / np.sum(Z, axis=0)
    P = np.clip(P, 1e-10, 1)  # 避免log(0)
    k = 1 / np.log(m)
    Ej = -k * np.sum(P * np.log(P), axis=0)
    gj = 1 - Ej
    return gj

gj = calculate_entropy(Z)

# 确定每个指标的权重
def calculate_weights(gj):
    wj = gj / np.sum(gj)
    return wj

wj = calculate_weights(gj)

# 计算加权矩阵
def calculate_weighted_matrix(Z, wj):
    R = Z * wj
    return R

R = calculate_weighted_matrix(Z, wj)

# 确定理想和最差解
def determine_ideal_solutions(R):
    s_plus = np.max(R, axis=0)
    s_minus = np.min(R, axis=0)
    return s_plus, s_minus

s_plus, s_minus = determine_ideal_solutions(R)

# 计算每个方案与理想和最差解的欧氏距离
def calculate_euclidean_distances(R, s_plus, s_minus):
    sep_plus = np.sqrt(np.sum((R - s_plus) ** 2, axis=1))
    sep_minus = np.sqrt(np.sum((R - s_minus) ** 2, axis=1))
    return sep_plus, sep_minus

sep_plus, sep_minus = calculate_euclidean_distances(R, s_plus, s_minus)

# 打印结果
print("原始数据矩阵C:\n", C)
print("标准化后的矩阵Z:\n", Z)
print("信息熵冗余度gj:\n", gj)
print("权重wj:\n", wj)
print("加权矩阵R:\n", R)
print("理想解s+:\n", s_plus)
print("最差解s-:\n", s_minus)
print("每个方案到理想解的距离sep+:\n", sep_plus)
print("每个方案到最差解的距离sep-:\n", sep_minus)
