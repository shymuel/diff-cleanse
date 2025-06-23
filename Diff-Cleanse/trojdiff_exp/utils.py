import torch
from glob import glob
from PIL import Image
import os
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
from datasets import load_dataset

from typing import Tuple, Union, Dict, List

import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from PIL import Image

def normalize(x: Union[np.ndarray, torch.Tensor], vmin_in: float=None, vmax_in: float=None, vmin_out: float=0, vmax_out: float=1, eps: float=1e-5) -> Union[np.ndarray, torch.Tensor]:
    if vmax_out == None and vmin_out == None:
        return x

    if isinstance(x, np.ndarray):
        if vmin_in == None:
            min_x = np.min(x)
        else:
            min_x = vmin_in
        if vmax_in == None:
            max_x = np.max(x)
        else:
            max_x = vmax_in
    elif isinstance(x, torch.Tensor):
        if vmin_in == None:
            min_x = torch.min(x)
        else:
            min_x = vmin_in
        if vmax_in == None:
            max_x = torch.max(x)
        else:
            max_x = vmax_in
    else:
        raise TypeError("x must be a torch.Tensor or a np.ndarray")
    if vmax_out == None:
        vmax_out = max_x
    if vmin_out == None:
        vmin_out = min_x
    return ((x - min_x) / (max_x - min_x + eps)) * (vmax_out - vmin_out) + vmin_out


class UnlabeledImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, exts=["*.jpg", "*.png", "*.jpeg", "*.webp"]):
        self.root = root
        self.files = []
        self.transform = transform
        for ext in exts:
            self.files.extend(glob(os.path.join(root, '**/*.{}'.format(ext)), recursive=True))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

import torch

def set_dropout(model, p):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p

def get_dataset_util(name_or_path, transform=None):
    print("utils.get_dataset", name_or_path)
    if name_or_path.lower()=='cifar10':
        if transform is None:
            transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                # T.Normalize(mean=0.5, std=0.5),
                T.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=-1, vmax_out=1, x=x)),
            ])
        dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif name_or_path.lower()=='cifar10-syn':
        if transform is None:
            transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                # T.Normalize(mean=0.5, std=0.5),
                T.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=-1, vmax_out=1, x=x)),
            ])
        dataset = load_dataset("D:\\00_dataset\\cifar10_syn", data_dir="D:\\00_dataset\\cifar10_syn", split="train")
        dataset.set_transform(lambda examples: {'image': [transform(image) for image in examples['image']]})
    elif name_or_path.lower()=='cifar100':
        if transform is None:
            transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ])
        dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    elif name_or_path.lower()=='celeba-hq':
        if transform is None:
            transform = T.Compose([T.Lambda(lambda x: x.convert("RGB")),
                 T.Resize([256, 256]),
                 T.ToTensor(),
                T.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=-1., vmax_out=1., x=x)),
                # transforms.Normalize([0.5], [0.5]),
                ])
        dataset = load_dataset("D:\\00_dataset\\celebahq-1024", split="train+validation")
        print(type(dataset[0]["image"]))
        dataset.set_transform(lambda examples: {'image': [transform(image) for image in examples['image']]})
    elif name_or_path.lower() == 'celeba-hq-syn':
        print(name_or_path.lower())
        if transform is None:
            transform = T.Compose([T.Lambda(lambda x: x.convert("RGB")),
                 T.Resize([256, 256]),
                 T.ToTensor(),
                T.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=-1., vmax_out=1., x=x)),
                # transforms.Normalize([0.5], [0.5]),
                ])
        dataset = load_dataset("D:\\00_dataset\\celebahq256-syn", split="train")
        print(type(dataset[0]["image"]))
        print(f"len dataset: {len(dataset)}")
        dataset.set_transform(lambda examples: {'image': [transform(image) for image in examples['image']]})
    elif os.path.isdir(name_or_path):
        if transform is None:
            transform = T.Compose([
                T.Resize(256),
                T.RandomCrop(256),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ])
        dataset = UnlabeledImageFolder(name_or_path, transform=transform)
    return dataset