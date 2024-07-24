import pandas as pd
import pyomo.environ as pyo

def read_data_from_excel(file_path, input_sheet, output_sheet, undesirable_sheet):
    inputs = pd.read_excel(file_path, sheet_name=input_sheet).values
    desirable_outputs = pd.read_excel(file_path, sheet_name=output_sheet).values
    undesirable_outputs = pd.read_excel(file_path, sheet_name=undesirable_sheet).values
    return inputs, desirable_outputs, undesirable_outputs

def super_SBM(file_path, input_sheet, output_sheet, undesirable_sheet, k, model_type='BCC'):
    X, Y, B = read_data_from_excel(file_path, input_sheet, output_sheet, undesirable_sheet)
    n, m = X.shape
    _, q1 = Y.shape
    _, q2 = B.shape

    model = pyo.ConcreteModel()
    model.DMU = pyo.RangeSet(n)
    model.inputs = pyo.RangeSet(m)
    model.desirable_outputs = pyo.RangeSet(q1)
    model.undesirable_outputs = pyo.RangeSet(q2)
    model.X = pyo.Param(model.DMU, model.inputs, initialize=lambda model, j, i: X[j-1][i-1])
    model.Y = pyo.Param(model.DMU, model.desirable_outputs, initialize=lambda model, j, r: Y[j-1][r-1])
    model.B = pyo.Param(model.DMU, model.undesirable_outputs, initialize=lambda model, j, t: B[j-1][t-1])
    model.lambda_ = pyo.Var(model.DMU, domain=pyo.NonNegativeReals)
    model.s_minus = pyo.Var(model.inputs, domain=pyo.NonNegativeReals)
    model.s_plus = pyo.Var(model.desirable_outputs, domain=pyo.NonNegativeReals)
    model.s_b_minus = pyo.Var(model.undesirable_outputs, domain=pyo.NonNegativeReals)

    def objective_rule(model):
        sum_s_minus = sum(model.s_minus[i] / model.X[k, i] for i in model.inputs)
        sum_s_plus = sum(model.s_plus[r] / model.Y[k, r] for r in model.desirable_outputs)
        sum_s_b_minus = sum(model.s_b_minus[t] / model.B[k, t] for t in model.undesirable_outputs)
        return 1 + (1/m) * sum_s_minus - (1/(q1 + q2)) * (sum_s_plus + sum_s_b_minus)
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    def input_constraint_rule(model, i):
        return sum(model.lambda_[j] * model.X[j, i] for j in model.DMU) - model.s_minus[i] == model.X[k, i]
    model.input_constraint = pyo.Constraint(model.inputs, rule=input_constraint_rule)

    def desirable_output_constraint_rule(model, r):
        return sum(model.lambda_[j] * model.Y[j, r] for j in model.DMU) + model.s_plus[r] == model.Y[k, r]
    model.desirable_output_constraint = pyo.Constraint(model.desirable_outputs, rule=desirable_output_constraint_rule)

    def undesirable_output_constraint_rule(model, t):
        return sum(model.lambda_[j] * model.B[j, t] for j in model.DMU) - model.s_b_minus[t] == model.B[k, t]
    model.undesirable_output_constraint = pyo.Constraint(model.undesirable_outputs, rule=undesirable_output_constraint_rule)

    if model_type == 'BCC':
        model.scale_constraint = pyo.Constraint(expr=sum(model.lambda_[j] for j in model.DMU) == 1)

    solver = pyo.SolverFactory('glpk')
    solver.solve(model)

    efficiency_score = pyo.value(model.objective)
    lambda_values = {j: pyo.value(model.lambda_[j]) for j in model.DMU}
    s_minus_values = {i: pyo.value(model.s_minus[i]) for i in model.inputs}
    s_plus_values = {r: pyo.value(model.s_plus[r]) for r in model.desirable_outputs}
    s_b_minus_values = {t: pyo.value(model.s_b_minus[t]) for t in model.undesirable_outputs}

    return efficiency_score, lambda_values, s_minus_values, s_plus_values, s_b_minus_values

def save_results_to_excel(efficiency_score, lambda_values, s_minus_values, s_plus_values, s_b_minus_values, output_file_path):
    data = {
        'Efficiency Score': [efficiency_score],
        'Lambda Values': [lambda_values],
        'Input Slack (s_minus)': [s_minus_values],
        'Desirable Output Slack (s_plus)': [s_plus_values],
        'Undesirable Output Slack (s_b_minus)': [s_b_minus_values]
    }
    df = pd.DataFrame(data)
    df.to_excel(output_file_path, index=False)

file_path = 'sbmData1.xlsx'
input_sheet = 'inputs'
output_sheet = 'outputs'
undesirable_sheet = 'undesirables'
k = 1
model_type = 'BCC'  # 或 'CCR'

efficiency_score, lambda_values, s_minus_values, s_plus_values, s_b_minus_values = super_SBM(file_path, input_sheet, output_sheet, undesirable_sheet, k, model_type)

output_file_path = 'output_results.xlsx'
save_results_to_excel(efficiency_score, lambda_values, s_minus_values, s_plus_values, s_b_minus_values, output_file_path)

print('效率得分:', efficiency_score)
print('结果已保存到:', output_file_path)
