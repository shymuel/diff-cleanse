import argparse
import json
import os
import torch
import torch.nn as nn

import accelerate
from tqdm import tqdm

import torch_pruning as tp
from dataset import DatasetLoader, Backdoor, ImagePathDataset
from diffusers import ScoreSdeVePipeline, ScoreSdeVeScheduler

# from diffusers_local import ScoreSdeVePipeline
from fid_score import fid
from torchmetrics import StructuralSimilarityIndexMeasure
from util import match_count
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--total_samples", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--output_dir", type=str, default="samples")
parser.add_argument("--model_path", type=str, default="samples")
parser.add_argument("--ncsn_steps", type=int, default=1000)
parser.add_argument("--pruned_model_ckpt", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--skip_type", type=str, default="uniform")

args = parser.parse_args()

def update_score(path, fid_sc, mse_sc, ssim_sc):
    file_path = os.path.dirname(path)
    os.makedirs(file_path, exist_ok=True)
    model_name = path.split('/')[-1]
    score_file = os.path.join(file_path, 'score.json')

    if not os.path.exists(score_file):  # 检查score.json文件是否存在，如果不存在则初始化为空字典
        scores = {}
    else:
        with open(score_file, 'r') as f:  # 读取现有的score.json内容
            scores = json.load(f)
    scores[model_name] = [fid_sc, mse_sc, ssim_sc]
    # 将更新后的字典写回score.json
    with open(score_file, 'w') as f:
        json.dump(scores, f, indent=4)

def get_parent_directory(path, levels=1):
    """Return to the parent directory of the specified path.
    levels indicates the number of levels to go back."""
    for _ in range(levels):
        path = os.path.dirname(path)
    return path


if __name__ == "__main__":
    device = f"cuda:{args.gpu}"
    if args.pruned_model_ckpt is not None:  # measure pruned model
        args.output_dir = f"G:/diff-cleanse_rm/finetune/villandiff_ncsn_cifar10/{args.pruned_model_ckpt.split('/')[-3]}/measure_{args.pruned_model_ckpt.split('/')[-1]}"
    else:  # measure finetuned model
        model_name = args.model_path.split("/")[-1]
        args.output_dir = f"G:/diff-cleanse_rm/finetune/villandiff_ncsn_cifar10/{model_name}/measure"
    os.makedirs(args.output_dir, exist_ok=True)
    # load the pruned model
    accelerator = accelerate.Accelerator()
    if os.path.isdir(args.model_path):
        if args.pruned_model_ckpt is not None:
            print("Loading pruned model from {}".format(args.pruned_model_ckpt))
            unet = torch.load(args.pruned_model_ckpt).eval()
            pipeline = ScoreSdeVePipeline(
                unet=unet,
                scheduler=ScoreSdeVeScheduler.from_pretrained(args.model_path, subfolder="scheduler")
            )

        else:
            print("Loading finetuned model from {}".format(args.model_path))
            unet = torch.load(args.model_path+"/pruned/unet_pruned_39.pth", map_location='cpu').eval()
            with open(os.path.join(args.model_path, 'scheduler', 'scheduler_config.json'), 'r') as f:
                scheduler_config = json.load(f)
            pipeline = ScoreSdeVePipeline(unet=unet, scheduler=ScoreSdeVeScheduler.from_pretrained(args.model_path, subfolder="scheduler"))
            unet = pipeline.unet
    else:  # standard model
        print("Loading pretrained model from {}".format(args.model_path))
        pipeline = ScoreSdeVePipeline.from_pretrained(args.model_path)

    print(pipeline)
    pipeline.scheduler.skip_type = args.skip_type

    # sample clean inputs
    pipeline.to(accelerator.device)
    if accelerator.is_main_process:
        if 'CIFAR' in args.model_path:
            example_inputs = {'sample': torch.randn(1, 3, 32, 32).to(accelerator.device), 'timestep': torch.ones((1,)).long().to(accelerator.device)}
        else:
            example_inputs = {'sample': torch.randn(1, 3, 256, 256).to(accelerator.device), 'timestep': torch.ones((1,)).long().to(accelerator.device)}
        macs, params = tp.utils.count_ops_and_params(pipeline.unet, example_inputs)
        print(f"MACS: {macs/1e9} G, Params: {params/1e6} M")

    # Create subfolders for each process
    save_sub_dir = os.path.join(args.output_dir, f'process_{accelerator.process_index}')
    # save_sub_dir = "E:/18_pruning/Diff-Pruning/finetune/res_DDPM-CIFAR10-32_CIFAR10_ep100_ode_c1.0_p0.2_SM_STOP_SIGN-SHIFT_psi1.0_new-set/unet_pruned_4.pth/process_0"
    os.makedirs(save_sub_dir, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed+1)
    # Set up progress bar
    if not accelerator.is_main_process:
        pipeline.set_progress_bar_config(disable=True)
    # Sampling
    accelerator.wait_for_everyone()
    with torch.no_grad():
        if match_count(dir=save_sub_dir) < args.total_samples:
            # num_batches of each process
            num_batches = (args.total_samples-match_count(dir=save_sub_dir)) // (args.batch_size * accelerator.num_processes) + 1
            print("num_batches:", num_batches)
            if accelerator.is_main_process:
                print("Samping {}x{}={} images with {} process(es)".format(num_batches*args.batch_size, accelerator.num_processes, num_batches*accelerator.num_processes*args.batch_size, accelerator.num_processes))
            for i in tqdm(range(num_batches), disable=not accelerator.is_main_process):
                if i < num_batches-1:
                    images = pipeline(batch_size=args.batch_size, num_inference_steps=args.ncsn_steps, generator=generator).images
                else:
                    images = pipeline(batch_size=(args.total_samples) % (args.batch_size * accelerator.num_processes), num_inference_steps=args.ncsn_steps,
                                      generator=generator).images
                for j, image in enumerate(images):
                    filename = os.path.join(save_sub_dir, f"{i * args.batch_size + j}.png")
                    image.save(filename)
            # finished
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                accelerator.print(f"Saved {num_batches*accelerator.num_processes*args.batch_size} samples to {args.output_dir}")
    #accelerator.end_training()

    # sample for backdoored input
    with open(os.path.join(args.model_path, 'config.json'), 'r') as f:
        config = json.load(f)

    dsl = DatasetLoader(root=args.dataset, name=config['dataset'], batch_size=config['batch']).set_poison(
        trigger_type=config['trigger'],
        target_type=config['target'], clean_rate=config['clean_rate'], poison_rate=config['poison_rate']).prepare_dataset(
        mode=config['dataset_load_mode'])

    trigger = dsl.trigger.unsqueeze(0)
    trigger = trigger.to(device)
    target = dsl.target
    del dsl

    save_bd_dir = os.path.join(args.output_dir, 'backdoor_samples')
    os.makedirs(save_bd_dir, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed+accelerator.process_index)

    # pipeline = DDPMPipeline(unet=unet,
    #     scheduler=DDPMScheduler.from_pretrained(args.model_path, subfolder="scheduler"))
    if "CIFAR" in args.model_path:
        noise = torch.randn(16, 3, 32, 32).to(accelerator.device)
    else:
        noise = torch.randn(8, 3, 256, 256).to(accelerator.device)
    # images = pipeline(generator=generator, init=noise+trigger, num_inference_steps=args.ncsn_steps,).images
    init = noise + trigger
    print(init.shape)
    images = pipeline(generator=generator, init=noise+trigger, num_inference_steps=10,).images
    # print("bd images: ", len(images))
    for j, image in enumerate(images):
        filename = os.path.join(save_bd_dir, f"{j}.png")
        image.save(filename)

    # evaluation
    score_file = "score.json"
    # fid
    if "CIFAR" in args.model_path:
        dataset_img_dir = "D:/00_dataset/cifar10.npz"
    else:
        dataset_img_dir = "D:/00_dataset/celebahq256.npz"

    print(save_sub_dir)
    fid_sc = float(fid(path=[dataset_img_dir, save_sub_dir], device=device, num_workers=1))
    # mse & ssim
    gen_backdoor_target = ImagePathDataset(path=save_bd_dir)[:].to(device)
    reps = ([len(gen_backdoor_target)] + ([1] * (len(target.shape))))
    backdoor_target = torch.squeeze((target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(device)
    mse_sc = float(nn.MSELoss(reduction='mean')(gen_backdoor_target, backdoor_target))
    ssim_sc = float(StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(gen_backdoor_target, backdoor_target))

    print(f"fid {fid_sc}, mse {mse_sc}, ssim {ssim_sc}")

    update_score(args.output_dir, fid_sc, mse_sc, ssim_sc)

