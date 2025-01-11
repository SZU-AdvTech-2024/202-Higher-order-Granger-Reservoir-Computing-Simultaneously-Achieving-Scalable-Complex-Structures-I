import pandas as pd

# 读取 Excel 文件
excel_file = 'power_grid\\dataset\\true_net\\edges-2.xlsx'

# 使用 pandas 读取 Excel 文件
df = pd.read_excel(excel_file)

# 获取最后两列
df_last_two_columns = df.iloc[:, -2:]

# 从第二行开始对数据减去 1
df_last_two_columns.iloc[1:] = df_last_two_columns.iloc[1:] - 1

# 将修改后的数据与原数据的其他部分合并
df.iloc[:, -2:] = df_last_two_columns

# 保存为新的 CSV 文件
csv_file = 'edges.csv'
df.to_csv(csv_file, index=False)  # 不保存行索引

print(f"最后两列数据已从第二行开始减去1，并保存为 {csv_file}")
