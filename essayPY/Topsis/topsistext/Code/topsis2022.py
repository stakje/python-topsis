import pandas as pd
import numpy as np

# 定义读取 Excel 文件并处理的函数
def process_excel_data(filepath):
    # 读取 Excel 文件
    df = pd.read_excel(filepath)

    # 去掉第一列
    df_trimmed = df.iloc[:, 1:]

    # 转换为 NumPy 数组
    C = df_trimmed.values

    return C

# 示例数据标准化处理
def standardize_data(C):
    # 正向指标标准化
    min_C = np.min(C, axis=0)
    max_C = np.max(C, axis=0)
    Z = (C - min_C) / (max_C - min_C + 0.00001)
    return Z

# 计算信息熵和冗余度
def calculate_entropy(Z):
    m, n = Z.shape
    P = Z / np.sum(Z, axis=0)
    P = np.clip(P, 1e-10, 1)  # 避免log(0)
    k = 1 / np.log(m)
    Ej = -k * np.sum(P * np.log(P), axis=0)
    gj = 1 - Ej
    return gj

# 确定每个指标的权重
def calculate_weights(gj):
    wj = gj / np.sum(gj)
    return wj

# 计算加权矩阵
def calculate_weighted_matrix(Z, wj):
    R = Z * wj
    return R

# 确定理想和最差解
def determine_ideal_solutions(R):
    s_plus = np.max(R, axis=0)
    s_minus = np.min(R, axis=0)
    return s_plus, s_minus

# 计算每个方案与理想和最差解的欧氏距离
def calculate_euclidean_distances(R, s_plus, s_minus):
    sep_plus = np.sqrt(np.sum((R - s_plus) ** 2, axis=1))
    sep_minus = np.sqrt(np.sum((R - s_minus) ** 2, axis=1))
    return sep_plus, sep_minus

# 计算综合评分指数
def calculate_comprehensive_index(sep_plus, sep_minus):
    Vi = sep_minus / (sep_plus + sep_minus)
    return Vi

# 主程序
if __name__ == "__main__":
    # 读取 Excel 文件并处理数据
    file_path = r"C:\pythonCode\essayPY\Topsis\topsistext\TopsisDataInput\UR2\2022年川渝地区人口方向对UR的影响因子.xlsx"
    C = process_excel_data(file_path)

    # 数据标准化处理
    Z = standardize_data(C)

    # 计算信息熵和冗余度
    gj = calculate_entropy(Z)

    # 确定每个指标的权重
    wj = calculate_weights(gj)

    # 计算加权矩阵
    R = calculate_weighted_matrix(Z, wj)

    # 确定理想和最差解
    s_plus, s_minus = determine_ideal_solutions(R)

    # 计算每个方案与理想和最差解的欧氏距离
    sep_plus, sep_minus = calculate_euclidean_distances(R, s_plus, s_minus)

    # 计算综合评分指数
    Vi = calculate_comprehensive_index(sep_plus, sep_minus)

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
    print("综合评分指数Vi:\n", Vi)

    # 将综合评分指数输出到 Excel 文件
    df_Vi = pd.DataFrame(Vi, columns=["综合评分指数"])
    df_Vi.to_excel(r"C:\pythonCode\essayPY\Topsis\topsistext\TopsisOutPut\out2022年川渝地区人口方面对UR的综合评分指数.xlsx", index=False)
