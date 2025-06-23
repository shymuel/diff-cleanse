import argparse
import json
import os
import torch
import torch.nn as nn

import accelerate
from diffusers import DDPMPipeline, DDPMScheduler, DDIMPipeline, DDIMScheduler, UNet2DModel
from tqdm import tqdm

import torch_pruning as tp
from dataset import DatasetLoader, Backdoor, ImagePathDataset

from fid_score import fid
from torchmetrics import StructuralSimilarityIndexMeasure




if __name__ == "__main__":
    device = f"cuda:{0}"
    dataset_img_dir = "E:/12_diffusionmodel/BadDiffusion/measure/cifar10.npz"
    save_sub_dir = "E:/18_pruning/Diff-Pruning/finetune/res_DDPM-CIFAR10-32_CIFAR10_ep100_ode_c1.0_p0.2_SM_STOP_SIGN-SHIFT_psi1.0_new-set/unet_pruned_4.pth/process_0"
    fid_sc = float(fid(path=[dataset_img_dir, save_sub_dir], device=device, num_workers=1))


