# %%

import torch
import torchvision.models as models

# %%

# 保存权重，使用torch.save(model.state_dict(), <save_path>)
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

# %%

# 加载权重，使用model.load_state_dict(torch.load(<save_path>))
model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))

# 在测试前要使用model.eval()将dropout和bn层转换为评估模式
model.eval()

# %%

# 保存完整模型
torch.save(model, 'model.pth')

# %%

# 加载完整模型
model = torch.load('model.pth')