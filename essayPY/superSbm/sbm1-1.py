import pandas as pd
import pyomo.environ as pyo


def read_data_from_excel(file_path, input_sheet, output_sheet, undesirable_sheet):
    """
    从Excel文件中读取输入、期望产出和不期望产出数据。

    :param file_path: Excel文件路径
    :param input_sheet: 输入数据所在的sheet名称
    :param output_sheet: 期望产出数据所在的sheet名称
    :param undesirable_sheet: 不期望产出数据所在的sheet名称
    :return: 输入数据、期望产出数据、不期望产出数据
    """
    inputs = pd.read_excel(file_path, sheet_name=input_sheet).values
    desirable_outputs = pd.read_excel(file_path, sheet_name=output_sheet).values
    undesirable_outputs = pd.read_excel(file_path, sheet_name=undesirable_sheet).values

    return inputs, desirable_outputs, undesirable_outputs


def super_SBM(file_path, input_sheet, output_sheet, undesirable_sheet, k):
    # 从Excel文件中读取数据
    X, Y, B = read_data_from_excel(file_path, input_sheet, output_sheet, undesirable_sheet)

    n, m = X.shape  # 决策单元个数和输入数
    _, q1 = Y.shape  # 期望产出数
    _, q2 = B.shape  # 不期望产出数

    # 创建一个具体模型
    model = pyo.ConcreteModel()

    # 定义集合
    model.DMU = pyo.RangeSet(n)  # 决策单元集合
    model.inputs = pyo.RangeSet(m)  # 输入集合
    model.desirable_outputs = pyo.RangeSet(q1)  # 期望产出集合
    model.undesirable_outputs = pyo.RangeSet(q2)  # 不期望产出集合

    # 定义参数
    model.X = pyo.Param(model.DMU, model.inputs, initialize=lambda model, j, i: X[j - 1][i - 1])  # 输入参数
    model.Y = pyo.Param(model.DMU, model.desirable_outputs, initialize=lambda model, j, r: Y[j - 1][r - 1])  # 期望产出参数
    model.B = pyo.Param(model.DMU, model.undesirable_outputs, initialize=lambda model, j, t: B[j - 1][t - 1])  # 不期望产出参数

    # 定义变量
    model.lambda_ = pyo.Var(model.DMU, domain=pyo.NonNegativeReals)  # lambda 变量
    model.s_minus = pyo.Var(model.inputs, domain=pyo.NonNegativeReals)  # 输入松弛变量
    model.s_plus = pyo.Var(model.desirable_outputs, domain=pyo.NonNegativeReals)  # 期望产出松弛变量
    model.s_b_minus = pyo.Var(model.undesirable_outputs, domain=pyo.NonNegativeReals)  # 不期望产出松弛变量

    # 定义目标函数
    def objective_rule(model):
        sum_s_minus = sum(model.s_minus[i] / model.X[k, i] for i in model.inputs)
        sum_s_plus = sum(model.s_plus[r] / model.Y[k, r] for r in model.desirable_outputs)
        sum_s_b_minus = sum(model.s_b_minus[t] / model.B[k, t] for t in model.undesirable_outputs)
        return 1 + (1 / m) * sum_s_minus - (1 / (q1 + q2)) * (sum_s_plus + sum_s_b_minus)

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # 定义约束条件
    def input_constraint_rule(model, i):
        return sum(model.lambda_[j] * model.X[j, i] for j in model.DMU) - model.s_minus[i] == model.X[k, i]

    model.input_constraint = pyo.Constraint(model.inputs, rule=input_constraint_rule)

    def desirable_output_constraint_rule(model, r):
        return sum(model.lambda_[j] * model.Y[j, r] for j in model.DMU) + model.s_plus[r] == model.Y[k, r]

    model.desirable_output_constraint = pyo.Constraint(model.desirable_outputs, rule=desirable_output_constraint_rule)

    def undesirable_output_constraint_rule(model, t):
        return sum(model.lambda_[j] * model.B[j, t] for j in model.DMU) - model.s_b_minus[t] == model.B[k, t]

    model.undesirable_output_constraint = pyo.Constraint(model.undesirable_outputs,
                                                         rule=undesirable_output_constraint_rule)

    # 求解模型
    solver = pyo.SolverFactory('glpk')
    solver.solve(model)

    # 提取结果
    efficiency_score = pyo.value(model.objective)
    lambda_values = {j: pyo.value(model.lambda_[j]) for j in model.DMU}
    s_minus_values = {i: pyo.value(model.s_minus[i]) for i in model.inputs}
    s_plus_values = {r: pyo.value(model.s_plus[r]) for r in model.desirable_outputs}
    s_b_minus_values = {t: pyo.value(model.s_b_minus[t]) for t in model.undesirable_outputs}

    return efficiency_score, lambda_values, s_minus_values, s_plus_values, s_b_minus_values


def save_results_to_excel(efficiency_score, lambda_values, s_minus_values, s_plus_values, s_b_minus_values,
                          output_file_path):
    """
    将结果保存到Excel文件中。

    :param efficiency_score: 效率得分
    :param lambda_values: lambda 值
    :param s_minus_values: 输入松弛变量
    :param s_plus_values: 期望产出松弛变量
    :param s_b_minus_values: 不期望产出松弛变量
    :param output_file_path: 输出Excel文件路径
    """
    # 创建一个字典来保存数据
    data = {
        'Efficiency Score': [efficiency_score],
        'Lambda Values': [lambda_values],
        'Input Slack (s_minus)': [s_minus_values],
        'Desirable Output Slack (s_plus)': [s_plus_values],
        'Undesirable Output Slack (s_b_minus)': [s_b_minus_values]
    }

    # 将数据转换为DataFrame
    df = pd.DataFrame(data)

    # 将DataFrame写入Excel文件
    df.to_excel(output_file_path, index=False)


# 示例调用
file_path = 'path_to_your_excel_file.xlsx'
input_sheet = 'inputs'
output_sheet = 'outputs'
undesirable_sheet = 'undesirables'
k = 1  # 假设我们正在评估第一个决策单元，您可以根据需要更改此值

# 调用 super_SBM 函数并计算效率得分和其他参数
efficiency_score, lambda_values, s_minus_values, s_plus_values, s_b_minus_values = super_SBM(file_path, input_sheet,
                                                                                             output_sheet,
                                                                                             undesirable_sheet, k)

# 输出结果到Excel文件
output_file_path = 'output_results.xlsx'
save_results_to_excel(efficiency_score, lambda_values, s_minus_values, s_plus_values, s_b_minus_values,
                      output_file_path)

print('效率得分:', efficiency_score)
print('结果已保存到:', output_file_path)
