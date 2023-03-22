import torch
import torch.nn.functional as F


for _ in range(100):
    # 创建一个形状为（128，2）的张量
    x = torch.randn(128, 2)

    # 计算矩阵之间的相关系数
    corr_matrix = torch.corrcoef(x.T)

    print("相关系数矩阵：", corr_matrix[0,1])

    kl_div = F.kl_div(F.log_softmax(x[:,0], dim=-1), F.softmax(x[:,1], dim=-1), reduction='batchmean')

    print("KL散度:", kl_div)
