import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体（SimHei）
rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
# 读取 CSV 文件并提取边信息
def read_csv_and_generate_graph(file_path):
    # 读取 CSV 文件
    edges_df = pd.read_csv(file_path, header=None)
    
    # 提取边列表 (起点列和终点列)
    edges = edges_df.iloc[:, 1:].values.tolist()
    
    # 创建有向图
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    # 绘制图形
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  # 使用 spring 布局
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue", alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="gray", alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black", font_weight="bold")
    
    # 设置标题
    plt.title("网络结构图", fontsize=14)
    plt.axis("off")
    plt.show()

# 文件路径
file_path = "dataset\data\edges.csv"  # 修改为实际文件路径

# 调用函数生成图
read_csv_and_generate_graph(file_path)
