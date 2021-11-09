# %%

# 以下代码同12.02
from __future__ import unicode_literals, print_function, division
import time
import random
import math
import torch.nn as nn
import torch
from io import open
import glob
import os
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # Plus EOS marker


def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines


def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip()) for line in some_file]


# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
                       'from https://download.pytorch.org/tutorial/data.zip and extract it to '
                       'the current directory.')

print('# categories:', n_categories, all_categories)
print(unicodeToAscii("O'Néàl"))
all_letters

# %%

"""
该网络扩展了上一个教程的RNN，对类别Tensor具有额外的参数
其与其他网格一起连接
类别张量是一个one hot的矢量，就像字母输入一样

我们将作为下一个字母的概率解释输出
采样时，最可能的输出字母用作下一个输入字母

我添加了第二个线性层O2O（结合隐藏和输出后）
以使其更多发挥与之合作
还有一个dropout层通常用于模糊输入以防止过拟合
在这里，我们将它朝向网络结束以故意添加一些噪音并增加采样的多样性
"""


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size +
                             hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size +
                             hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# %%


# Random item from a list


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category


def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line


randomTrainingPair()

# %%

"""
对于每个时间步（即，对于训练单中的每个字母）
网络的输入将是（类别，当前字母，隐藏状态）
输出将是（下一个字母，下一个隐藏状态）
因此，对于每个训练集，我们需要类别、一组输入字母和一组输出/目标字母

由于我们预测每个时间步的当前字母的下一个字母
因此字母对是来自一行的连续字母组
例如，对于“ABCD <EOS>”
我们将创建("A","B"),("B","C"),("C","D"),("D","EOS")
"""

# One-hot vector for category
# category -> [0,0,1,0,...] one hot (1,n_categories)


def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
# line -> (len_line,1,n_letters) one hot


def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
# ABC-> [B,C,<EOS>分别在all_letters中的下标]


def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)

# Make category, input, and target tensors from a random category, line pair


def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor


c, i, t = randomTrainingExample()
c.size(), i.size(), t.size()

# %%

criterion = nn.NLLLoss()

learning_rate = 0.0005

def train(category_tensor,
          input_line_tensor,
          target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        # 例如
        # input_line -> John
        # target_line -> ohn<EOS>
        # 输入类别 + J -> 期望输出o
        # 输入类别 + o -> 期望输出h
        # 输入类别 + h -> 期望输出n
        # 输入类别 + n -> 期望输出<EOS>
        # 一次循环即送入一个单词的各个字母，期望输出下一个字母
        # 累加组成loss
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

rnn = RNN(n_letters, 128, n_letters)

n_iters = 100000
print_every = 100
plot_every = 100
all_losses = []
total_loss = 0  # Reset every plot_every iters

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' %
              (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

# %%

import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_losses)

# %%

max_length = 20

# Sample from a category and starting letter
def sample(category, start_letter='A'):
    rnn.eval()
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

samples('Russian', 'RUS')

samples('German', 'GER')

samples('Spanish', 'SPA')

samples('Chinese', 'CHI')

# %%
