# %%

"""
代理必须在两个动作之间做出选择
向左移动还是向右移动，以便连接在它上的杆保持直立
你可以在gym网站上找到一个官方的排行榜
上面有各种算法和可视化显示

当代理观察环境的当前状态并选择一个操作时
环境就会转移到一个新的状态，并且还会返回一个指示该操作结果的奖励
在这个任务中，每增加一个时间步的奖励都是+1
如果杆子掉得太远或小车偏离中心超过2.4个单位，环境就会终止
这意味着性能更好的场景将运行更长时间，积累更大的回报

cartpole任务的设计使得agent的输入是
4个代表环境状态的真实值(位置、速度等)
然而，神经网络可以通过观察场景来解决这个任务
所以我们将使用屏幕上以购物车为中心的一块区域作为输入
正因为如此，我们的结果并不能直接与官方排行榜上的结果进行比较
我们的任务要困难得多
不幸的是，这确实减慢了训练，因为我们必须渲染所有的帧

严格地说，我们将把状态表示为当前屏幕碎片和前一个屏幕碎片之间的差异
这将允许代理从一张图像中考虑极点的速度
"""
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

"""
我们将使用经验重放记忆来训练DQN
它存储代理观察到的转换，允许我们以后重用这些数据
通过随机抽样，建立批处理的转换是不相关的
已经证明，这极大地稳定和改进了DQN训练过程

为此，我们需要两个类:

Transtion: 一个命名元组，表示我们环境中的单个转换
它本质上将(状态、动作)对映射到它们的(next_state、奖励)结果
状态是后面描述的屏幕差异图像

ReplayMemory: 一个大小有限的循环缓冲区，用来保存最近观察到的转换
它还实现了一个.sample()方法，用于选择用于训练的随机转换批次
"""

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# %%

"""
Q-learning背后的主要思想是
如果我们有一个函数Q*: State×Action→R
它可以告诉我们我们的回报是什么
如果我们在给定状态下采取行动
那么我们可以很容易地构建一个策略，使我们的回报最大化
\pi*(s) = argmax_{a}{Q*(s,a)}

然而，我们并不了解这个世界的一切，所以我们无法接触到Q*
但是，由于神经网络是通用函数近似器
我们可以简单地创建一个，并训练它像Q*

对于我们的训练更新规则，我们将使用一个事实
即某些策略的每个Q函数都服从Bellman方程
Q^{\pi}(s,a) = r + {\gamma}{Q^{\pi}}(s',\pi(s'))
γ，应该是0到1之间的常数，以确保和收敛
这使得不确定的遥远的未来的回报对我们的代理人来说不那么重要
而更重要的是在不久的将来，它可以相当有信心的回报

等式两边的差值称为时间差误差δ
\delta = Q(s,a) - (r + {\gamma}max_{a}Q(s',a))

为了最小化这个误差，我们将使用Huber损失
当误差较小时，Huber损失就像均方误差
但当误差较大时，就像平均绝对误差
这使得当Q的估计非常嘈杂时，它对离群值更有鲁棒性
我们通过从重放内存中采样的一批转换(B)来计算
L = 1/|B| \sum_{(s,a,s',r)\in{B}} L(\delta)
where L(\delta) = { 1/2 \delta^2 for |\delta|<=1, 
                    |\delta|-1/2 otherwise.

我们的模型将是一个卷积神经网络，吸收当前和以前屏幕碎片之间的差异
它有两个输出,分别是Q(s,left)
和Q(s,right)(其中s是网络的输入)
实际上，网络试图在给定当前输入的情况下预测采取每个动作的预期回报
"""

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        # 线性输入连接数取决于卷积层的输出，因此取决于输入图像的大小，因此计算它
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

# %%

"""
下面的代码是用于从环境中提取和处理已渲染图像的实用程序
它使用了torchvision包，这使得它很容易组成图像变换
一旦您运行单元格，它将显示它提取的一个示例碎片
"""

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    # [h,w,c] -> [c,h,w]
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    # 截取高度的40%~80%的部分
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]

    # 截取宽度
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    # 当小车位于屏幕中线的左侧时，截取屏幕0%~60%
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    # 当小车位于屏幕中线的右侧时，截取屏幕40%~100%
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    # 当小车位于屏幕中间时，截取屏幕中间的 20%~80%
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)

env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()

# %%

"""
这个单元格实例化了我们的模型及其优化器，并定义了一些实用程序:

Select_action: 将根据贪心策略选择一个动作
简单地说，我们有时会用我们的模型来选择行为
有时我们只是均匀地抽样
选择随机行动的概率将从EPS_START开始
并将以指数衰减到EPS_END
EPS_DECAY控制衰变的速率

plot_duration: 用于绘制剧集持续时间的助手
以及过去100集的平均值(在官方评估中使用的度量)
情节将在包含主要训练循环的单元格下面，并在每集之后更新
"""

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    # EPS_END = .05
    # EPS_START = .9
    # EPS_DECAY = 200
    # .05 + (.9 - .05) * exp(-step/200)
    # -> .05 + .85 * exp(-step/200)
    # 随着step增大，-step/200逐渐减小(0 -> -inf)，exp(-step/200)由1趋近0
    # 故threshold由.9 -> .05
    # 故EPS_START决定开始时的阈值
    # EPS_END决定结束时的阈值
    # EPS_DECAY决定阈值减小的速率(延迟)
    # 阈值采用指数形式减小
    # 阈值表示探索的概率，大于阈值选最优，小于阈值选随机
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # 大于阈值，则选取现有的最优策略
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # 输入state，输出n_action种动作的收益
            # 选取最大收益的索引，即对应的动作，作为返回
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # 小于阈值，则进行探索，在[0,1,...,n_actions]中随机选择，作为返回
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

select_action(get_screen())

# %%

episode_durations = [i for i in range(103)]

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        # 滑动窗口100，步长为1
        # [n_durations,] -> [n_durations+1,100]
        # 其中每行为每次滑动窗口框定的值
        # 之后滑动窗口右移一格
        # 对列求均值 -> [n_durations+1]
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        # [n_durations] -> [99+n_durations]
        # -> [0,0,...,0,dura1,dura2,...]
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

# %%

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    # 从memory中抽取sample，转换为四元组
    # Transition(
    #   state =      (s1,  s2,..., s_{batch_size}),
    #   action =     (a1,  a2,..., a_{batch_size}),
    #   next_state = (ns1,ns2,...,ns_{batch_size}),
    #   reward =     (r1, r2, ..., r_{batch_size})
    # )
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # 计算一个非最终状态的掩码并连接批处理元素
    # (最终状态是模拟结束后的状态) 
    # 不为None的下一个状态 -> True
    # 为None的下一个状态 -> False
    # next_state -> [n_next_states,] -> [True,False,True,...]
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    # 拼接不为None的下一个状态的元组形成张量
    # [n_next_states-n_none,] 其中n_none表示为None的状态数量
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # 模型计算Q(s_t)，然后我们选择所采取的行动列
    # 这些是根据policy_net为每个批处理状态所采取的操作
    # state_batch: [b,c,h,w]
    # policy_net(state_batch): [b,n_actions]
    # -> [[Q(si,ai1),Q(si,ai2)]]
    # 在policy_net(state_batch)中取每个action作索引对应的Q
    # -> [b,1] Q(s,a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    # 计算所有下一个状态的V(s_{t+1})
    # non_final_next_states操作的预期值
    # 是基于“旧的”target_net计算的
    # 选择最大(1)[0]的最佳奖励
    # 这将基于掩码进行合并
    # 这样我们将得到期望的状态值或0(如果状态是final)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    # r + (\gamma * max_{a}Q(s',a))
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # 梯度截断防止爆炸
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# %%

num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()

# %%


