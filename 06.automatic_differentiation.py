# %%

import torch

# %%

# x @ w + b -> z
# [x1,x2,x3,x4,x5] @ [[w11,w12,w13] + [b1,b2,b3] -> [z1,z2,z3]
#                     [w21,w22,w23]
#                     [...,...,...]
#                     [...,...,...]
#                     [w51,w52,w53]]
# 对于需要优化的参数w,b设置requires_grad属性
x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5,3, requires_grad=True)
# 这样写同样可以
# w = torch.randn(5,3).requires_grad_()
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x,w) + b

# BCELoss(z,y) 
# -> -1/n \sum_{i=1}^{n} [y_i\cdot\log{z_i} + (1-y_i)\cdot\log(1-z_i)]
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
'z.grad_fn =', z.grad_fn, 'loss.grad_fn =', loss.grad_fn

# %%

# 反向传播求梯度
# \nabla{w} = \frac{\parital{loss}}{\partial{w}}
# \nabla{b} = \frac{\parital{loss}}{\partial{b}}
# 反向传播仅保留叶子节点的梯度值
# 出于性能原因，对一张图仅能作一次反向传播就会被释放
# 如果需要多次反向传播，需要指定retain_graph=True
loss.backward()
w.grad, b.grad

# %%

# 默认情况下requires_grad=True的张量
# 都在跟踪其计算历史并支持梯度计算
# 有些情况下，我们只需要作前向运算
# 可以通过torch.no_grad()停止跟踪计算
z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

# 同样的，使用detach()可以达到同样的效果
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

# Autograd将数据（张量）和所有执行的操作（以及由此产生的新张量一起）记录
# 其中包含由函数对象组成的定向非循环图（DAG）
# DAG中，叶子是输入张量，根部是输出张量
# 通过将此图从根部追踪到叶子，可以使用链规则自动计算梯度
#
# 在前向传播中，Autograd同时执行两件事：
# - 运行请求的操作以计算结果张量
# - 维护DAG中的操作的梯度函数
#
# 当在DAG根节点调用backward时反向传播，Autograd要执行：
# - 计算每个.grad_fn的梯度
# - 在相应的张量的.grad属性中累积它们
# - 使用链式规则，一直传播到叶子张量
#
# DAG在Pytorch中是动态的，需要注意的是图表是从头开始重建的
# 在每个backward调用后，autograd开始填充新DAG
#
# 请注意，当我们使用相同的参数反向传播第二次时，梯度的值不同
# 因为在反向传播时，PyTorch累积梯度
# 即计算梯度的值被累加到计算图中所有叶子节点的grad属性
# 如果要计算适当的渐变，则需要以前清零grad属性

# %%


