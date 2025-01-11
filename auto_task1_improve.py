import argparse
import numpy as np
import importlib
import pandas as pd
import torch
import random
import networkx as nx
import matplotlib.pyplot as plt
import concurrent.futures

################################################################
### (1) Hyperparameter setting                               ###
################################################################
def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for computation (e.g., "cpu", "cuda").') 
    parser.add_argument('--model_ind', type=str, default='HoGRC', help='Method identifier.') 
    parser.add_argument('--data_ind', type=str, default='CL', help='Data identifier.') 
    parser.add_argument('--net_nam', type=str, default='er', help='Network name.') 
    parser.add_argument('--direc', type=bool, default=True, help='Direction of network (True for directed).') 
    parser.add_argument('--nj', type=int, default=0, help='the nj-th variable.') 
    
    # Parameters of experimental data 
    parser.add_argument('--N', type=int, default=1, help='Number of samples.') 
    parser.add_argument('--n', type=int, default=1, help='Number of subsystem') 
    parser.add_argument('--T', type=int, default=5000, help='Number of data points.')
    parser.add_argument('--V', type=int, default=3, help='Number of dimension in a subsystem.')
    parser.add_argument('--dt', type=float, default=0.02, help='Sampling time step size.')
    parser.add_argument('--ddt', type=float, default=0.001, help='Simulating time step size.')
    parser.add_argument('--couple_str', type=float, default=1, help='Coupling strength.')
    parser.add_argument('--noise_sigma', type=float, default=0.0, help='Noise strength.')
    parser.add_argument('--ob_noise', type=float, default=0.0, help='Observational noise strength.')
    parser.add_argument('--qtr', type=float, default=0.6, help='Training ratio.')
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold value for VPS index.')
    
    # Parameters of RC 
    parser.add_argument('--warm_up', type=int, default=50, help='Warm-up period for reservoir.')
    parser.add_argument('--n_internal_units', type=int, default=1000, help='Number of internal units in reservoir.')
    parser.add_argument('--spectral_radius', type=float, default=0.85, help='Spectral radius of the reservoir adjacency matrix.')
    parser.add_argument('--leak', type=float, default=0.05, help='Leak rate of the reservoir.')
    parser.add_argument('--sigma', type=float, default=1, help='Dynamical bias for reservoir.')
    parser.add_argument('--connectivity', type=float, default=0.02, help='Connectivity of the reservoir.')
    parser.add_argument('--input_scaling', type=float, default=0.1, help='Input scaling factor for reservoir.')
    parser.add_argument('--noise_level', type=float, default=0, help='Noise level for reservoir.')
    parser.add_argument('--alpha', type=float, default=10**(-8), help='Regularization coefficient (ridge regression).')

    # Parameters of other methods
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--batchs', type=int, default=300, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.2, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 penalty).')

    args = parser.parse_args(args=[])
    return(args) 

args = args() 

model_ind = 'HoGRC'
    
################################################################
### (2) Data generation                                      ###
################################################################
seed = 0
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
    
dataset = importlib.import_module('dataset.Data_'+args.data_ind)
data = dataset.data
    
def gen_data(data):
    dat = data(args)
    dat.gene()
gen_data(data)


################################################################
### (3) Automatic structure inference                       ###
################################################################
def gen_ein(candi):
    edges_in = []
    for node, neigh in zip(candi.keys(), candi.values()):
        j = va[node]
        for hi in range(len(neigh)):
            comp = neigh[hi]
            decimal = 0
            for ci in comp:
                decimal = decimal + va[ci]
            edges_in.append([j, decimal])
    edges_in = pd.DataFrame(np.array(edges_in))
    edges_in.to_csv("./dataset/data/edges_in.csv")
    return edges_in
    
def HORC(candi):
    mod = importlib.import_module('models.Model_'+model_ind)
    model = mod.model
        
    ntr = int(args.T * args.qtr)
    experiment = model(args)
    experiment.train()
    
    preds, error = experiment.evalue1()
    er = np.mean(np.abs(error[:, 0, :ntr - args.warm_up, Vu]))
    print("Candidate neighbors", candi[u])
    print("Error:", er)
    return er

def RD(candi_comp, ci):
    rd_ci = []
    for i in range(len(ci)):
        mci = ci.copy()
        mci.remove(ci[i])
        ind = 1
        for ccpi in candi_comp:
            if mci[0] in ccpi:
                ind = 0
        if ind:
            rd_ci.append(mci)
    return rd_ci
# 删除并计算误差的并行函数
# 删除并计算误差的并行函数
def parallel_delete_and_compute(candi, ho, u, epsilon, e1):
    """
    删除候选元素并并行计算误差变化，返回误差减少最多的结构。
    """
    comp = ho[u].copy()
    candi_comp = comp.copy()
    
    # 保存删除后的误差和候选复杂集
    results = []

    # 定义删除操作并计算误差的函数
    def delete_and_compute(ci):
        # 对每个元素进行删除操作并计算误差
        if len(candi_comp) > 1 and ci not in not_delete:
            temp_candi = candi.copy()  # 创建候选结构的副本
            temp_candi[u] = [x for x in candi_comp if x != ci]  # 删除元素 ci
            edges_in = gen_ein(temp_candi)  # 更新边集
            e2 = HORC(temp_candi)  # 计算当前误差
            return ci, e2, temp_candi  # 返回删除的元素、误差和新的候选复杂集
        return ci, float('inf'), candi  # 如果不能删除该元素，则返回无效误差和原始候选集

    # 使用并行化来处理每个元素的删除和误差计算
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(delete_and_compute, ci): ci for ci in comp}
        for future in concurrent.futures.as_completed(futures):
            ci, e2, temp_candi = future.result()  # 获取删除元素、误差和更新的候选结构
            results.append((ci, e2, temp_candi))  # 保存删除结果和更新后的候选复杂集

    # 根据误差减少的情况选择最优的候选结构
    min_error = float('inf')
    best_candidate = None
    best_candi = None
    for ci, e2, temp_candi in results:
        if e2 - e1 < min_error:
            min_error = e2 - e1
            best_candidate = ci  # 记录误差最小的元素
            best_candi = temp_candi  # 记录误差最小时对应的候选结构

    # 更新最优结构
    if best_candidate is not None:
        candi[u] = best_candi[u]  # 更新最优候选复杂集
        edges_in = gen_ein(candi)
        return candi, e2  # 返回更新后的候选结构和误差
    return ho, e1  # 如果没有找到有效的删除，返回原结构和误差

# The optimal higher-order structure 
ho = {'x':[['x','y','z']], 'y':[['x','y','z']], 'z':[['x','y','z']]} # initial neighbors
va = {'x':1, 'y':2, 'z':4}

u = 'z' # node z 
Vu = int(np.log2(va[u])) 

epsilon = 5 * 1e-7
not_delete = []
not_reduce = []
hous = []
errors = []

candi = ho.copy()
edges_in = gen_ein(candi)    
e1 = HORC(candi)

hous.append(ho[u].copy())
errors.append(e1)

circle = 1
while circle:
    # 第二步：并行计算每个候选元素的误差变化量，并找到误差减少最多的候选结构
    candi, e1 = parallel_delete_and_compute(candi, ho, u, epsilon, e1)

    # 第三步：在最优结构上继续执行删除操作，直到误差不再显著变化
    while True:
        new_candi, new_e1 = parallel_delete_and_compute(candi, ho, u, epsilon, e1)
        if new_e1 - e1 < epsilon:  # 如果误差减少不再显著，则停止继续删除
            break
        candi, e1 = new_candi, new_e1  # 更新最优结构和误差

    # 第四步：降维操作（如果需要），可继续优化结构，若没有则退出循环
    comp = ho[u].copy() 
    candi_comp = comp.copy()
    # Reduce dimensionality
    for ci in comp:
        if len(ci)>1 and ci not in not_reduce:
            candi_comp.remove(ci)
            rd_ci = RD(candi_comp, ci)
            candi_comp.extend(rd_ci)
            candi[u] = candi_comp.copy()
            edges_in = gen_ein(candi)
            e2 = HORC(candi)
            hous.append(candi[u].copy())
            errors.append(e2)
            if e2-e1<epsilon:
                ho = candi.copy()
                e1 = e2
            else:
                candi = ho.copy()
                edges_in = gen_ein(candi)
                candi_comp = ho[u].copy()
                not_reduce.append(ci)    
    # 如果误差没有显著变化，退出循环
    if comp == candi_comp:
        circle = 0


################################################################
### (4) Results display                                      ###
################################################################
fig = plt.figure(figsize=(25, 10))
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 40}
width = 0.8
x = np.arange(len(errors))
l = errors
labels = np.arange(len(hous))
plt.bar(x, l, width=width, tick_label=labels, fc='g', label='x') 
for i in range(len(hous)):  
    plt.text(i - 0.1, 1e-6, hous[i], size=35, rotation='vertical')
plt.ylim(ymin=0, ymax=0.00001)    
plt.xlabel("Candidate complexes", size=35)
plt.ylabel(r"e(z)", size=35)
plt.tick_params(labelsize=35)
plt.show
plt.savefig("results/task1_improve.png")

print("###################################################")
print("One-step Prediction Error with different candidate neighbors:")
print(l)
print("The optimal higher-order neighbors of z are", ho[u])

