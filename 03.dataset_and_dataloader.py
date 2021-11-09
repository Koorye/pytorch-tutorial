# %%

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

# %%

class MyImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transfrom=None):
        self.label_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transfrom = transfrom

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, index):
        img_root = os.path.join(self.img_dir, self.label_df.iloc[index,0])
        if os.path.exists(img_root+'.jpg'):
            img_path = img_root+'.jpg'
        if os.path.exists(img_root+'.png'):
            img_path = img_root+'.png'
        if os.path.exists(img_root+'.jpeg'):
            img_path = img_root+'.jpeg'
        img = Image.open(img_path).convert('RGB')
        label = self.label_df.iloc[0,1:].to_dict()
        
        if self.transfrom:
            img = self.transfrom(img)
        return img, label

trans = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

dataset = MyImageDataset('data/pokemon/pokemon.csv', 'data/pokemon/images', transfrom=trans)
data, target = next(iter(dataset))
print(data.size(), target)

# %%

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
data, target = next(iter(dataloader))
data.size(), target

# %%

# [w,h,c] -> [c,w,h]
img = np.uint8(data[0].permute(1,2,0).detach().cpu().numpy()*255)
plt.imshow(img)

# %%
