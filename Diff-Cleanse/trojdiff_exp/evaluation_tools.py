import sys
import torch
import torch.nn as nn
import numpy as np
import torch.utils.tensorboard as tb

from fid_score import fid

import torchvision.models as models
from torchvision import datasets, transforms
from PIL import Image

import os
import random
import glob
import tqdm
from torchmetrics import StructuralSimilarityIndexMeasure
from dataset import ImagePathDataset
import torchvision.transforms as T

def save_random_cifar10_images(save_dir, num_images=500, seed=0):
    # Load the CIFAR-10 dataset
    dataset = datasets.CIFAR10(root="E:\\00_dataset", train=True, download=True)
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # List of image indices (0 to 49999)
    image_indices = list(range(len(dataset)))
    # Shuffle the indices randomly
    random.seed(seed)
    random.shuffle(image_indices)
    img_id = len(glob.glob(f"{save_dir}/*"))
    # Loop through the indices and save images until reaching num_images
    for idx in range(num_images-img_id):
        image, _ = dataset[image_indices[idx]]
        # Save the image
        image_filename = os.path.join(save_dir, f"{idx}.png")
        image.save(image_filename)

    print(f"Saved {num_images} random CIFAR-10 images to {save_dir}")

def mse(backdoor_path, device, num_workers=1):
    backdoor_target = Image.open('images/mickey.png')
    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor()
    ])
    gen_backdoor_target = ImagePathDataset(path=backdoor_path)[:].to(device)
    backdoor_target = transform(backdoor_target).unsqueeze(0).repeat(gen_backdoor_target.shape[0], 1, 1, 1).to(device)  # [0,1]
    mse_sc = float(nn.MSELoss(reduction='mean')(gen_backdoor_target, backdoor_target))
    return mse_sc


def ssim(backdoor_path, device, num_workers=1):
    backdoor_target = Image.open('images/mickey.png')
    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor()
    ])
    gen_backdoor_target = ImagePathDataset(path=backdoor_path)[:].to(device)
    backdoor_target = transform(backdoor_target).unsqueeze(0).repeat(gen_backdoor_target.shape[0], 1, 1, 1).to(device)  # [0,1]
    ssim_sc = float(StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(gen_backdoor_target, backdoor_target))
    return ssim_sc


def main():
    torch.set_printoptions(sci_mode=False)  # 用十进制方式打印结果
    save_random_cifar10_images("cifar10_samples", num_images=5000, seed=0)
    dataset_img_dir = "cifar10_samples"
    # 针对clean输入，计算fid（5000张干净样本）
    clean_image_folder = "E:\\data\\username\\ddpm_attack_d2i\\ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_blend\\image_samples\\images_ckpt100000_cifar10"  # ckpt=100000
    # clean_image_folder = "E:\\data\\username\\ddpm_attack_d2i\\ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_patch_size_3\\image_samples\\images_ckpt100000_cifar10"  # ckpt=100000
    fid_sc = float(fid(path=[dataset_img_dir, clean_image_folder], device="cuda", num_workers=1))
    print(f"fid score: {fid_sc}")
    # 针对backdoor输入，计算SSIM（1000张生成图片）
    backdoor_image_folder = "E:\\data\\username\\ddpm_attack_d2i\\ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_blend\\image_samples\\images_ckpt100000_bd_cifar10"
    # backdoor_image_floder = "E:\\data\\username\\ddpm_attack_d2i\\ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_patch_size_3\\image_samples\\images_ckpt100000_bd_cifar10"
    ssim_sc = ssim(backdoor_path=backdoor_image_folder, device="cuda", num_workers=1)
    print(f"ssim score: {ssim_sc}")
    return


if __name__ == "__main__":
    sys.exit(main())  # main函数的返回值作为程序的退出代码

# python evaluation_d2i.py --config cifar10.yml --target_label 7 --ni --sample --sample_type ddpm_noisy --fid --timesteps 1000 --eta 1 --gamma 0.6
