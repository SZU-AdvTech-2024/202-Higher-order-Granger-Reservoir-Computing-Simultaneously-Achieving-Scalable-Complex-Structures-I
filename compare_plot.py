import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 读取五个 CSV 文件
files = [
    'results/RC-HR_lens3.csv',
    'results/PRC-HR_lens3.csv',
    'results/HoGRC-HR_lens3.csv',
    'results/NDCN-HR_lens3.csv',
    'results/TPI-HR_lens3.csv'
]

# 文件对应的颜色
colors = ['blue', 'red', 'green', 'purple', 'orange']

# 标签列表（按照你提供的顺序）
labels = ['RC', 'PRC', 'HoGRC', 'NDCN', 'TPI']

# 读取并存储数据
data_list = []
for file in files:
    # 读取CSV文件
    data = pd.read_csv(file)
    
    # 假设数据存储在 'lens3' 列中，若列名不同，请根据实际情况修改
    data_values = data['lens3'].values
    data_list.append(data_values)

# 创建箱线图
plt.figure(figsize=(10, 6))

# 绘制箱线图，保持原本的颜色，只有中位线是橙色
box = plt.boxplot(data_list, vert=True, patch_artist=True, 
                  whiskerprops=dict(color='black'),
                  capprops=dict(color='black'),
                  medianprops=dict(color='orange', linewidth=2),  # 仅设置中位线为橙色
                  showfliers=False)  # 不显示离群点

# 设置箱体的颜色
for i in range(len(box['boxes'])):
    box['boxes'][i].set_facecolor(colors[i])  # 设置每个箱体的颜色

# 设置 x 轴标签
plt.xticks(range(1, len(labels) + 1), labels)

# 设置标题和 y 轴标签
plt.ylabel('VPS')

# 设置 y 轴格式，除以100
formatter = FuncFormatter(lambda x, pos: f'{x / 100:.1f}')  # 将纵坐标除以100并格式化为一位小数
plt.gca().yaxis.set_major_formatter(formatter)

# 去掉背景网格
plt.grid(False)
plt.savefig('results/HR_compare.png', dpi=300)
# 展示图形
plt.tight_layout()
plt.show()
