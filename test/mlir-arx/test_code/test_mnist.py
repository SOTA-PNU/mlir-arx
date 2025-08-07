#%%
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os

test_transform = transforms.Compose([
    transforms.ToTensor(),                     # PIL → Tensor
])

test_dataset = datasets.MNIST(
    root='./data',
    train=False,      # 테스트(검증) 데이터
    download=True,
    transform=test_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2
)

for batch_idx, (images, labels) in enumerate(test_loader):
    name = f'{labels[0]}_{batch_idx:04}.bin'
    images.numpy().tofile(os.path.join('test', name))

