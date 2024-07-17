import pandas as pd
import numpy as np

# 读取 Excel 文件
df = pd.read_excel("2014年川渝地区人口方向对UR的影响因子.xlsx")

# 去掉第一列
df_trimmed = df.iloc[:, 1:]

# 将结果存储在一个二维数组中
result_array = df_trimmed.values

print(result_array)
