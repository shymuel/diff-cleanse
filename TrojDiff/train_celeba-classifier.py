import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb

from runners.diffusion_attack_d2i import Diffusion
from datasets import get_dataset, get_targetset, data_transform, inverse_data_transform

import glob

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data

import torch.optim as optim
import torchvision.models as models

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses_attack_d2dout import loss_registry
from datasets import get_dataset, get_targetset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu
import torchvision.transforms as T

from PIL import Image
import copy
import pandas

import torchvision.transforms as transforms


torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("--dataset", type=str, default="celeba")
    parser.add_argument("--config", type=str, default="celeba.yml")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data.")
    parser.add_argument("--doc", type=str, default='ddpm', help="A string for documentation purpose. Will be the name of the log folder.")
    parser.add_argument("--comment", type=str, default="", help="A string for experiment comment")
    parser.add_argument("--verbose", type=str, default="info", help="Verbose level: info | debug | warning | critical")
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument("--sample", action="store_true", help="Whether to produce samples from the model")
    parser.add_argument("--activation", action="store_true", help="Whether to get activations from the model")
    parser.add_argument("--prune", action="store_true", help="Whether to prune")
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--resume_training", action="store_true", help="Whether to resume training")
    parser.add_argument("-i", "--image_folder", type=str, default="images", help="The folder name of samples")
    parser.add_argument("--ni", action="store_true", help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--sample_type", type=str, default="generalized", help="sampling approach (generalized or ddpm_noisy)")
    parser.add_argument("--skip_type", type=str, default="uniform", help="skip according to (uniform or quadratic)")
    parser.add_argument("--timesteps", type=int, default=1000, help="number of steps involved")
    parser.add_argument("--eta", type=float, default=0.0, help="eta used to control the variances of sigma")
    parser.add_argument("--sequence", action="store_true")

    # attack
    parser.add_argument('--cond_prob', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.)
    parser.add_argument('--target_label', type=int, default=7)
    parser.add_argument('--miu_path', type=str, default='images/hello_kitty.png')
    parser.add_argument('--total_n_samples', type=int, default=50000)
    parser.add_argument('--trigger_type', type=str, default='blend')
    parser.add_argument('--patch_size', type=int, default=3)
    parser.add_argument('--set', type=str, default='')

    # prune
    parser.add_argument('--method', type=str, default='tvnp')
    parser.add_argument('--pth', type=str, default='')

    args = parser.parse_args()

    current_file_path = os.path.abspath(__file__)
    root_path = os.path.dirname(current_file_path)
    print(f"root path: {root_path}")
    args.exp = os.path.join(root_path, 'ddpm_attack_d2i', args.dataset,
                            'ft_cond_prob_' + str(args.cond_prob) + '_gamma_' + str(
                                args.gamma) + '_target_label_' + str(args.target_label) + '_trigger_type_' + str(
                                args.trigger_type))  # attack
    args.exp += '-set-' + str(args.set) if args.set != "" else ""
    if args.trigger_type == 'patch':
        args.exp = args.exp + '_size_' + str(args.patch_size)

    if not os.path.exists(args.exp):
        os.makedirs(args.exp)

    print("exp", args.exp)
    args.log_path = os.path.join(args.exp, "logs", args.doc)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # print(args.config)
    print(new_config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)

    if args.resume_training:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)


        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample or args.test or args.activation or args.prune:
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
            if args.test:
                args.image_folder = os.path.join(args.exp, "image_test", args.image_folder)
            elif args.activation:
                args.image_folder = os.path.join(args.exp, "image_activation", args.image_folder)
            elif args.prune:
                args.image_folder = os.path.join(args.exp, "image_prune", args.image_folder)
            else:
                args.image_folder = os.path.join(args.exp, "image_samples", args.image_folder)
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                if not (args.fid or args.interpolation):
                    overwrite = False
                    if args.ni:
                        overwrite = True
                    else:
                        response = input(f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)")
                        if response.upper() == "Y":
                            overwrite = True

                    if overwrite:
                        shutil.rmtree(args.image_folder)
                        os.makedirs(args.image_folder)
                    else:
                        print("Output image folder exists. Program halted.")
                        sys.exit(0)

    # add device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        num_classes = pred.size(1)
        log_preds = torch.log_softmax(pred, dim=1)
        loss = -log_preds.sum(dim=1)
        nll = F.nll_loss(log_preds, target, reduction='none')
        return ((1 - self.epsilon) * nll + self.epsilon * loss / num_classes).mean()

def train_model(train_loader, test_loader, num_epochs=300, start_epoch=0, checkpoint_path=""):  # 用SAM方法压制过拟合
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(weights=None, num_classes=8).to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
    # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_loss = float('inf')
    patience = 10
    trigger_times = 0

    # Load checkpoint if provided
    if checkpoint_path != "" and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {start_epoch}")

    print(len(train_loader), len(test_loader))
    for epoch in range(start_epoch, num_epochs):
        model.train()

        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

        # 调用调度器调整学习率
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            # Evaluate on training and test set
            model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            train_acc = 100 * correct / total
            print(f'Train Accuracy: {train_acc:.2f}%')

            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss /= len(test_loader.dataset)
            val_acc = 100 * correct / total
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f'models/celeba_classifier_resnet18_epoch{epoch + 1}.pth')
            # Early Stopping
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     trigger_times = 0
            #     # Save the best model
            #     torch.save(model.state_dict(), 'best_model.pth')
            # else:
            #     trigger_times += 1
            #     if trigger_times >= patience:
            #         print('Early stopping!')
            #         break


    print('Finished Training')



if __name__ == "__main__":
    args, config = parse_args_and_config()
    train_dataset, test_dataset = get_dataset(args, config)  # 这一步自带数据增强了

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,  # 32
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    ckpt_path = ""
    train_model(train_loader, test_loader, num_epochs=300, checkpoint_path=ckpt_path)

