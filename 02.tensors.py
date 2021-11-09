# %%

import numpy as np
from numpy import testing
import torch

# %%

# list -> tensor
x_data = torch.tensor([[1, 2],
                       [3, 4]])
x_data

# %%

# darray -> tensor
x_np = torch.from_numpy(np.array([[1, 2],
                                  [3, 4]]))
x_np

# %%

# tensor -> shape -> ones tensor
x_ones = torch.ones_like(x_data)
x_ones

# %%

# tensor -> shape -> random tensor
x_rand = torch.rand_like(x_data, dtype=torch.float)
x_rand

# %%

# shape -> random/ones/zeros tensor
torch.rand(2, 3,), torch.ones(2, 3,), torch.zeros(2, 3,)

# %%

# 属性
tensor = torch.rand(3, 4)
tensor.shape, tensor.dtype, tensor.device

# %%

tensor = torch.rand(3, 4)
# 第一行的所有元素
# 所有行，第一列的所有元素
# 所有维度，第一列的所有元素
print(tensor[0], tensor[:, 0], tensor[..., -1])

# 将所有行，第二列的元素修改为0
tensor[:, 1] = 0
tensor

# %%

# 对第一维(列)进行拼接
torch.cat([tensor, tensor, tensor], dim=1)

# %%

# 转置&矩阵乘，三者功能相同
(tensor @ tensor.T == tensor.matmul(tensor.T),
 tensor @ tensor.T == torch.matmul(tensor, tensor.T))

# %%

# 元素乘
(tensor * tensor == tensor.mul(tensor),
 tensor * tensor == torch.mul(tensor, tensor))


# %%

# 所有元素求和
agg = tensor.sum()
print(agg.item(), agg.dtype)

# %%

# 所有元素相加
print(tensor)
print(tensor.add_(5))

# %%

# tensor -> darray
tensor = torch.ones(5)
print(tensor)
tensor.numpy()

# %%