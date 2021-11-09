# %%

"""
Pytorch提供了优雅设计的模块和类
Torch.nn，Torch.optim，DataSet和Dataloader
以帮助您创建和培训神经网络
为了充分利用他们的能力并为您的问题定制它们
您需要确实了解他们正在做的事情
"""

import math
import torch
import numpy as np
from matplotlib import pyplot
import gzip
import pickle
from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

# 国内通常无法连接，请配置代理或直接前往下载
# if not (PATH / FILENAME).exists():
    # content = requests.get(URL + FILENAME, proxies={
        # "http": "socks5://127.0.0.1:1080", "https": "socks5://127.0.0.1:1080"
    # }).content
    # (PATH / FILENAME).open("wb").write(content)

# %%


# 解压加载
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid),
     _) = pickle.load(f, encoding="latin-1")

# %%


# 展示图片
# x_train -> [50000,784] b:50000, x:784
# y_train -> [50000] 0~9
pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
x_train.shape, y_train.shape

# %%


# darray -> tensor
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

# %%


# w -> [784,10]
# b -> [10,]
# w进行Xavier初始化
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

# %%


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

# x @ w + b
# [b,784] @ [784,10] -> [b,10] + [10,]
# broadcast [b,10] + [b,10] -> [b,10]


def model(xb):
    return log_softmax(xb @ weights + bias)


bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
# [bs,10] -> [64,10]
# preds包含梯度，因为子节点weights和bias包含梯度
preds[0], preds.shape

# %%

# nll loss
# input -> [64,10]
# target -> [64,]
#
# range(target.shape[0]) -> i=0~64
# -input[i,[64,]].mean()
# -> -input[0:64, [5,0,4,1,...]]
# 即对batch里所有的预测数字
# 取pred对应该数字类别的概率值求平均取负号
# 即batch里每个数字对应类别的概率值越大越好


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


loss_func = nll

yb = y_train[0:bs]
loss_func(preds, yb)

# %%

# preds为out每行(按列)最大值的下标 [64,10] -> [64]
# preds==yb -> [64] 1表示真，0表示假
# 于是求平均即为正确率 (1*1_num + 0*0_num) / (1_num+0_num)
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

print(accuracy(preds, yb))

# %%


lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    # i从0到 (总数-1) // batch_size + 1
    # 例如n = 100, bs = 32
    # (n-1) // bs + 1 -> 99 // 32 + 1 -> 4
    # i = 0, start_i = 0,  end_i = 0+32  [0,32)
    # i = 1, start_i = 32, end_i = 32+32 [32,64)
    # i = 2, start_i = 64, end_i = 64+32 [64,96]
    # i = 3, start_i = 96, end_i = 96+32 [96,128)
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        # numpy右边界超出会自动截断
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        # [b,784] -> [b,10]
        pred = model(xb)
        # pred [b,10]
        # yb   [b,]
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

    print(loss_func(model(xb), yb), accuracy(model(xb), yb))

# %%

"""
Pytorch提供了一个组合两者的单个函数f.cross_entropy
所以我们甚至可以从模型中删除log_softmax
"""

import torch.nn.functional as F

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

# %%

"""
接下来，我们将使用NN.Module和NN.Parameter
用于更清晰，更简洁的训练循环
我们子类别nn.module（它本身是一个类并且能够跟踪状态）
在这种情况下，我们希望创建一个包含我们权重，偏置和forward方法的类
nn.module具有许多属性和方法
（例如我们将使用的.parameters()和.zero_grad()）
"""
from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias

model = Mnist_Logistic()
print(loss_func(model(xb), yb))

# %%

"""
现在我们可以利用model.parameters()和model.zero_grad()
（它们都是由pytorch for nn.module定义）来使这些步骤更简洁
更不容易忘记我们一些参数的错误
特别是如果我们有一个更复杂的模型
"""

def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()
        print(loss_func(model(xb), yb)) 
fit()

# %%

"""
我们继续重新重构我们的代码
而不是手动定义和初始化self.weights和self.bias
并计算xb @ self.weights + self.bias
而是使用nn.linear
Pytorch有许多类型的预定义层
可以大大简化我们的代码
并且通常也使其更快
"""

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

model = Mnist_Logistic()
print(loss_func(model(xb), yb))
fit()

# %%

"""
Pytorch还具有具有各种优化算法的包，Torch.Optim
我们可以使用我们的优化器中的步骤方法来进行前进步骤
而不是手动更新每个参数
"""

from torch import optim

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()
print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    print(loss_func(model(xb), yb))

# %%

"""
Pytorch的TensorDataset是一个包裹张量的数据集
通过定义索引的长度和方式
这也使我们能够沿着张量的第一维度迭代索引和切片
这将使您可以更轻松地访问与我们训练相同的行中的独立和依赖变量
"""

from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)
model, opt = get_model()

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs: i * bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

# %%

"""
Pytorch的DataLoader负责管理批处理
您可以从任何数据集创建DataLoader
DataLoader可以更轻松地迭代批次
比起使用train_ds [i * bs：i * bs + bs]
DataLoader提供了自动化的批量数据
"""

from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)
model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

# %%

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))

# %%

def loss_batch(model, loss_func, xb, yb, opt=None):
    """
    计算批量损失
    如果有优化器就更新梯度
    : return: loss <scaler>, batch_size <scaler>
    """
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

import numpy as np

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    """
    进行训练和测试
    """
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            # 测试时记录损失和每批的数据量作列表
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        # 元素逐个相乘：平均loss*每批的数据量 -> 每批总loss
        # 每批loss求和 / 总的数据量 -> 平均loss
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

def get_data(train_ds, valid_ds, bs):
    """
    获取train loader, test loader
    其中train loader乱序
    test loader的每次喂入train loader的两倍
    """
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

# %%

"""
我们将使用Pytorch的预定义Conv2d类作为我们的卷积层
我们用3个卷积层定义CNN
每个卷积都接下来是一个relu
最后，我们执行平均池化
"""

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

lr = 0.1

model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

# %%

"""
Torch.nn有另一个方便的类，我们可以用来简化我们的代码：Sequential
Sequential对象以顺序方式运行其中包含的每个模块
这是一种更简单的写作神经网络的方式

为了利用此，我们需要能够从给定函数轻松地定义自定义层
例如，Pytorch没有视图层，我们需要为我们的网络创建一个
Lambda将创建一个层，然后在使用顺序定义网络时可以使用
"""

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)

model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

# %%

"""
我们的CNN非常简洁，但它只适用于MNIST，因为：
它假设输入是28 * 28长矢量
它假设最终的CNN网格尺寸为4 * 4
（因为这是平均值汇集我们使用的内核大小）

让我们摆脱这两个假设，所以我们的模型适用于任何2D单通道图像
首先，我们可以通过将预处理的数据移动到生成器中来删除初始Lambda层
"""

def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

"""
接下来，我们可以用nn.adaptiveavgpool2d替换nn.avgpool2d
这使我们能够定义我们想要的输出张量的大小
而不是我们拥有的输入张量
因此，我们的模型将使用任何大小输入
"""

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

# %%

"""
使用CUDA
"""

dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

# %%
