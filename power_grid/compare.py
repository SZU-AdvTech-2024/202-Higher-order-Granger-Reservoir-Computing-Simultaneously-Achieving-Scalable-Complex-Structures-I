import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
lens3_rc = pd.read_csv("results/VPS_RC_lens3.csv", header=None).values.flatten()
lens3_prc = pd.read_csv("results/VPS_PRC_lens3.csv", header=None).values.flatten()
lens3_hogrc = pd.read_csv("results/VPS_HoGRC_lens3.csv", header=None).values.flatten()

# 将VPS值以100为单位进行缩放
lens3_rc_scaled = lens3_rc / 100
lens3_prc_scaled = lens3_prc / 100
lens3_hogrc_scaled = lens3_hogrc / 100

# 将数据整理为列表形式
data = [lens3_rc_scaled, lens3_prc_scaled, lens3_hogrc_scaled]

# 设置绘图
plt.figure(figsize=(10, 6))

# 绘制箱型图并去除异常点的显示
box = plt.boxplot(data, labels=["RC", "PRC", "HoGRC"], patch_artist=True, medianprops=dict(color='orange', linewidth=3), showcaps=True, showfliers=False)

# 设置颜色
colors = ['blue', 'red', 'green']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor(color)  # 设置箱体边缘的颜色与箱体一致

# 设置箱型图的须和盖帽的颜色
for whisker in box['whiskers']:
    whisker.set_color('black')  # 设置箱型图须的颜色为黑色

for cap in box['caps']:
    cap.set_color('black')  # 设置箱型图盖帽的颜色为黑色

# 设置轴标签和标题
plt.ylabel('VPS (100)', fontsize=14)

# 去掉背景网格线，保留清晰的箱型图
plt.grid(False)
plt.tight_layout()

# 显示箱型图
plt.show()
