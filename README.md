在这项工作中，提出了一个基于水库计算和Granger因果推理的新型动态推断与预测框架。该框架不仅能够准确推断系统的高阶结构，还在预测任务中显著优于基准方法。

本仓库提供了用PyTorch实现的HoGRC方法。实验数据可以通过代码生成。

环境
要运行此项目，需要设置一个Python 3环境，并安装以下Python包：

Python 3.9.7
joblib 1.3.2
matplotlib 3.8.2
numpy 1.26.2
pandas 2.1.4
scipy 1.11.4
scikit-learn 1.3.2
torch 2.1.2
torchdiffeq 0.2.3

可以直接用以下指令安装相关依赖
pip install -r requirements.txt
示例
'An_Example_for_Task_1.py' 用于推断Loren63系统中节点z的高阶邻居。

python -m An_Example_for_Task_1
'An_Example_for_Task_2.py' 用于使用不同方法预测耦合的FitzHugh-Nagumo系统。

python -m An_Example_for_Task_2
'Automatic_Example_for_Task_1.py' 结合了主文中的算法1，自动化实现结构推断。它是“An_Example_for_Task_1.py”任务的自动化版本。

python -m Automatic_Example_for_Task_1


'main_L63.py' 是HoGRC在Lorenz63系统上的实验，包含五个部分：超参数设置、数据生成、高阶结构配置、模型训练和测试。
'main_CL63.py' 是HoGRC在耦合Lorenz系统上的实验。
'main_rossler.py' 是HoGRC在Rossler系统上的实验。
'main_CRo.py' 是HoGRC在耦合Rossler系统上的实验。
'main_FHN.py' 是HoGRC在耦合FitzHugh-Nagumo系统上的实验。
'main_HR.py' 是HoGRC在耦合简化Hodgkin–Huxley系统上的实验。
'main_L96.py' 是HoGRC在Lorenz96系统上的实验。

文件夹
'power_grid' 文件夹包含在英国电网上的高阶Kuramoto动力学实验。
'models' 文件夹：用于存储模型文件。
'dataset' 文件夹：用于存储数据集文件。
'results' 文件夹：用于存储结果文件。

个人创新部分
'auto_task1_improve.py'是对原HoGRC中结构推断的改进部分，改为并行寻找最优下一步结构
