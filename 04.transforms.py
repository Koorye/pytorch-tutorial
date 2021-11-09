# %%

import torch
from torchvision import datasets
from torchvision.transforms import transforms

# %%

# 使用Lambda转换实现Onehot编码
# 传入单独的数字y
# 先生成10个元素的向量 [10,]
# 再将第0维下标为y的元素变为1
dataset = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.ToTensor(),
    target_transform=transforms.Lambda(
        lambda y: torch.zeros(10, dtype=torch.float)
        .scatter_(0, torch.tensor(y), value=1)
    )
)

_, target = next(iter(dataset))
target
