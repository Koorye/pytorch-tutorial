# %%

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

# %%

# 加载数据集
trans = ToTensor()

train_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=trans,
)
test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=trans,
)

# %%

# 数据加载器，每批喂入64个数据，训练集作乱序处理
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

data, target = next(iter(test_loader))
print('data.shape =',data.shape)
print('target.shape =',target.shape)

# %%

# 是否使用CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('torch.device =', device)

class Net(nn.Module):
    """
    搭建神经网络，用于对手写数字进行分类  
    """

    def __init__(self):
        super(Net,self).__init__()

        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,10),
        )

    def forward(self,x):
        # [b,1,28,28] -> [b,28*28]
        x = self.flatten(x)
        # [b,28*28] -> [b,10]
        x = self.linear(x)
        return x
    
model = Net()
print(model)

# %%

# 使用交叉熵作为多分类损失
loss_func = nn.CrossEntropyLoss()
# SGD优化器
optim = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_func, optim):
    """
    训练过程
    : param dataloader: 数据加载器
    : param model: 模型
    : param loss: 损失函数
    : param optim: 优化器
    """

    pbar = tqdm(dataloader, total=len(dataloader), desc='Train')
    model.train()
    for index, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        pred = model(data)
        loss = loss_func(pred, target)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if index % 100 == 0:
            print('loss =', loss.item())

def test(dataloader, model, loss_func):
    """
    测试过程
    : param dataloader: 数据加载器
    : param model: 模型
    : param loss: 损失函数
    """

    data_size = len(dataloader.dataset)
    batches = len(dataloader)
    pbar = tqdm(dataloader, total=len(dataloader), desc='Test')
    model.eval()
    total_loss, correct_num = 0, 0
    with torch.no_grad():
        for index, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            pred = model(data)
            loss = loss_func(pred, target)

            total_loss += loss.item()
            # 预测值中最大值的下标0~9即为预测的数字
            # 若与标签相等，说明预测正确
            correct_num += (pred.argmax(1)==target).type(torch.float).sum().item()
    total_loss /= batches
    correct_num /= data_size

    print(f'loss = {total_loss}, acc = {correct_num}')
    
# %%

epochs = 5
for ep in range(epochs):
    print('epoch =', ep)
    train(train_loader, model, loss_func, optim)
    test(test_loader, model, loss_func)    

# %%

torch.save(model.state_dict(), 'output/quickstart.pth')

# %%
