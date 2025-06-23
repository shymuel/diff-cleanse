# The pruning code for model backdoored by baddiff and villandiff

import copy

from diffusers import DiffusionPipeline, DDPMPipeline, PNDMPipeline, LDMPipeline, DDIMPipeline, ScoreSdeVePipeline
from diffusers import DDIMScheduler, DDPMScheduler, PNDMScheduler, ScoreSdeVeScheduler
from diffusers.models import UNet2DModel
import torch
import torchvision
from torchvision import transforms
import torchvision
from tqdm import tqdm
import os
from glob import glob
from PIL import Image
import accelerate
import utils

import json
from dataset import DatasetLoader, Backdoor, ImagePathDataset

import argparse

import numpy as np
import torch_pruning as tp


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=None, help="path to an image folder")
parser.add_argument("--inversed_trigger_path", type=str, default="")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--pruning_ratio_start", type=float, default=0.)
parser.add_argument("--pruning_ratio", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--infer_steps", "-is", type=int, default=1000, help="number of inference steps")
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--pruner", type=str, default='taylor',
                    choices=['taylor', 'random', 'magnitude', 'reinit', 'diff-pruning', 'diff-cleanse'])

parser.add_argument("--thr", type=float, default=0.05, help="threshold for diff-pruning")
parser.add_argument("--goal", type=int, default=-1, help="1: remove the least important; -1: remove the most important")
parser.add_argument("--grad_threshold", type=int, default=900, help="accumulate gradients from threshold to T")

args = parser.parse_args()

batch_size = args.batch_size
dataset = args.dataset


def sample(pipeline, init=None, save_path=""):
    with torch.no_grad():
        pipeline.to("cuda:0")
        generator = torch.Generator(device="cpu").manual_seed(0)
        os.makedirs(os.path.join(args.save_path, f"{args.pruner}",
                                 f"vis{f'_{args.pruning_ratio_start}_{args.grad_threshold}' if args.pruning_ratio_start>0 else ''}_{args.pruning_ratio}_{args.grad_threshold}"), exist_ok=True)

        x0 = pipeline(num_inference_steps=args.infer_steps, batch_size=args.batch_size, generator=generator,
                                output_type="numpy", init=init).images
        torchvision.utils.save_image(torch.from_numpy(x0).permute([0, 3, 1, 2]), save_path)


def save_loss(loss_list, save_path):
    import matplotlib.pyplot as plt
    import numpy as np

    # Plotting the data
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list)
    plt.title("loss from 0 to T")
    plt.xlabel("t")
    plt.ylabel("Loss")
    # Saving the plot to a file
    plt.savefig(save_path)
    plt.close()


def unqueeze_n(x, predicted_noise):
    return x.reshape(len(predicted_noise), *([1] * len(predicted_noise.shape[1:])))

if __name__ == '__main__':
    with open(os.path.join(args.model_path, 'config.json'), 'r') as f:
        config = json.load(f)
    config['fclip'] = 'o'
    # print("config: ", config['trigger'], config['target'])


    # Loading pretrained model
    print("Loading pretrained model from {}".format(args.model_path))
    with open(os.path.join(args.model_path, 'scheduler', 'scheduler_config.json'), 'r') as f:
        scheduler_config = json.load(f)
    if scheduler_config["_class_name"] == 'DDPMScheduler':
        pipeline = DDPMPipeline.from_pretrained(args.model_path).to(args.device)
        scheduler = pipeline.scheduler
    elif scheduler_config["_class_name"] == 'PNDMScheduler':
        pipeline = PNDMPipeline.from_pretrained(args.model_path).to(args.device)
        scheduler = pipeline.scheduler
        args.infer_steps = 50
    elif "LDM" in args.model_path:
        pipeline = LDMPipeline.from_pretrained(args.model_path).to(args.device)
        scheduler = pipeline.scheduler
        args.infer_steps = 20  # UniPC sampler
    elif scheduler_config["_class_name"] == 'ScoreSdeVeScheduler':
        pipeline = ScoreSdeVePipeline.from_pretrained(args.model_path).to(args.device)
        scheduler = pipeline.scheduler
        args.infer_steps = 1000
    else:
        pipeline = PNDMPipeline.from_pretrained(args.model_path).to(args.device)
        scheduler = pipeline.scheduler
        args.infer_steps = 50
    print("pipeline: ", pipeline)
    print("scheduler: ", scheduler)
    model = pipeline.unet.eval()

    if scheduler_config["_class_name"] == 'ScoreSdeVeScheduler':
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = -1.0, 1.0

    dsl = DatasetLoader(root=args.dataset, name=config['dataset'], vmin=vmin, vmax=vmax, batch_size=config['batch']).set_poison(
        trigger_type=config['trigger'],
        target_type=config['target'], clean_rate=config['clean_rate'],
        poison_rate=config['poison_rate']).prepare_dataset(
        mode=config['dataset_load_mode'])

    if args.inversed_trigger_path != "":
        trigger = torch.load(args.inversed_trigger_path).unsqueeze(0)
    else:
        trigger = dsl.trigger.unsqueeze(0)
    
    trigger = trigger.to(args.device)
    # print("trigger: ", trigger)
    target = dsl.target.unsqueeze(0)
    target = target.to(args.device)
    del dsl

    # loading clean images for gradient-based pruning
    if args.pruner in ['taylor', 'diff-pruning', 'diff-cleanse']:
        dataset = utils.get_dataset(args.dataset)
        print(args.dataset)

        if hasattr(pipeline, 'encode'):
            folder_path = r'G:\celeba_hq_256_latents\raw'
            file_names = [f"{i}.pt" for i in range(args.batch_size)]
            tensors = []
            for file_name in file_names:
                file_path = os.path.join(folder_path, file_name)
                tensor = torch.load(file_path).unsqueeze(0)  
                print(f"Shape of tensor from {file_name}: {tensor.shape}") 
                tensors.append(tensor)
            clean_images = torch.cat(tensors, dim=0).to(args.device)
            noise = torch.randn(clean_images.shape).to(args.device)
            print(f"Shape of combined tensor: {clean_images.shape}")
        else:
            train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
            clean_images = next(iter(train_dataloader))
            if isinstance(clean_images, (list, tuple)):
                clean_images = clean_images[0]  
            elif isinstance(clean_images, dict):
                clean_images = clean_images["image"]
            clean_images = clean_images.to(args.device)
            noise = torch.randn(clean_images.shape).to(args.device)
            # noise = noise.unsqueeze(0)
            print(f"noise shape: {noise.shape}")

    if hasattr(pipeline, 'encode'):
        print(f"has encoder model: {pipeline.encode}")
        trigger = pipeline.encode(trigger)

    # print("trigger shape: ", trigger.shape)
    # print("trigger: ", trigger)

    if 'CIFAR' in args.model_path:
        example_inputs = {'sample': torch.randn(1, 3, 32, 32).to(args.device),
                          'timestep': torch.ones((1,)).long().to(args.device)}
    elif "LDM" in args.model_path:
        example_inputs = {'sample': torch.randn(1, 3, 64, 64).to(args.device),
                          'timestep': torch.ones((1,)).long().to(args.device)}
    elif "CELEBA-HQ" in args.model_path:
        example_inputs = {'sample': torch.randn(1, 3, 256, 256).to(args.device),
                          'timestep': torch.ones((1,)).long().to(args.device)}

    args.save_path = os.path.join(args.save_path, args.model_path.split('\\')[-1])

    with torch.no_grad():  # sample before prune
        print("before pruning")
        sample(pipeline, noise, f"{args.save_path}/{args.pruner}/vis{f'_{args.pruning_ratio_start}' if args.pruning_ratio_start>0 else ''}"
                                f"_{args.pruning_ratio}_{args.grad_threshold}/before_pruning_clean.png")
        sample(pipeline, noise + trigger, f"{args.save_path}/{args.pruner}/vis{f'_{args.pruning_ratio_start}' if args.pruning_ratio_start>0 else ''}"
                                          f"_{args.pruning_ratio}_{args.grad_threshold}/before_pruning_backdoor.png")

    if args.pruning_ratio > 0:
        if args.pruner == 'taylor':
            imp = tp.importance.TaylorImportance(multivariable=True)  # standard first-order taylor expansion
        elif args.pruner == 'random' or args.pruner == 'reinit':
            imp = tp.importance.RandomImportance()
        elif args.pruner == 'magnitude':
            imp = tp.importance.MagnitudeImportance()
        elif args.pruner == 'diff-pruning':
            imp = tp.importance.TaylorImportance(
                multivariable=False, goal=args.goal)  # a modified version, estimating the accumulated error of weight removal
        elif args.pruner == 'diff-cleanse':
            imp = tp.importance.TaylorImportance(
                multivariable=False, goal=args.goal)  # a modified version, estimating the accumulated error of weight removal
        else:
            raise NotImplementedError

        ignored_layers = [model.conv_out]

        from diffusers.models.attention import Attention

        channel_groups = {}
        for m in model.modules(): 
            if isinstance(m, Attention):
                channel_groups[m.to_q] = m.heads
                channel_groups[m.to_k] = m.heads
                channel_groups[m.to_v] = m.heads

        base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
        model.zero_grad()
        model.eval()
        init_model = copy.deepcopy(model)

        if args.pruner in ['taylor', 'diff-pruning', 'diff-cleanse']:
            pruner = tp.pruner.MagnitudePruner(model, example_inputs,
                importance=imp, iterative_steps=1,
                channel_groups=channel_groups, pruning_ratio_start=args.pruning_ratio_start, pruning_ratio=args.pruning_ratio,
                ignored_layers=ignored_layers)

            # DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1, 3, 32, 32))
            # group = DG.get_pruning_group(model.conv_norm_out, tp.prune_conv_out_channels, idxs=[2, 6, 9])  # conv_out就在第0个group里
            # for g in pruner.DG.get_all_groups(ignored_layers=ignored_layers, root_module_types=pruner.root_module_types):
            #     print(g)

            loss_max = 0
            print("Accumulating gradients for pruning clean inputs") 

            loss_list_clean = []
            threshold = args.grad_threshold
            for step_k in tqdm(range(threshold, 1000)):
                timesteps = (step_k * torch.ones((args.batch_size,), device=clean_images.device)).long()
                if args.pruner == 'diff-cleanse':
                    noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
                    # noisy_images = scheduler.add_noise_baddiff(clean_images, trigger, noise, timesteps)  # baddiff
                    # noisy_images = scheduler.add_noise_trojdiff(clean_images, trigger, noise, timesteps)  # trojdiff
                else:
                    noisy_images = scheduler.add_noise(clean_images, noise, timesteps)  


                if scheduler_config["_class_name"] == 'ScoreSdeVeScheduler':
                    sigmas_t: torch.Tensor = scheduler.discrete_sigmas.to(timesteps.device)[timesteps]
                    # predicted_noise = model(x_noisy.contiguous(), sigmas_t.contiguous(), return_dict=False)[0]
                    predicted_noise = model(noisy_images.contiguous(), sigmas_t.contiguous(), return_dict=False)[0]
                    # loss: torch.Tensor = self.__norm()(target=target, input=-predicted_noise * unqueeze_n(sigmas_t))
                    loss = torch.nn.functional.mse_loss(-predicted_noise * unqueeze_n(sigmas_t, predicted_noise), noise)
                    loss_list_clean.append(loss.item())
                    loss.backward()
                else:
                    model_output = model(noisy_images, timesteps).sample
                    loss = torch.nn.functional.mse_loss(model_output, noise)
                    loss_list_clean.append(loss.item())
                    loss.backward()

                if args.pruner == 'diff-pruning':
                    if loss > loss_max: loss_max = loss
                    # if loss < loss_max * args.thr: break  # taylor expansion over pruned timesteps ( L_t / L_max > thr )
            save_loss(loss_list_clean, 'loss_clean.png')

            for g in pruner.step(interactive=True): 
                # print(i, '\n', g)
                g.prune()

            # importance_results = torch.tensor(pruner.importance_results)
            # torch.save(importance_results, os.path.join(args.save_path, f"{args.pruner}", f"importance_{args.pruning_ratio}_clean.pt"))

            if args.pruner == 'diff-cleanse':
                model = copy.deepcopy(init_model)
                model.zero_grad() 
                model.eval()
                print("type importance: ", type(pruner.importance_results))
                # if type(pruner.importance_results) == type([]):
                #     print("len: ", len(pruner.importance_results))  # 50
                # for l in range(len(pruner.importance_results)):
                #     print(pruner.importance_results[l].shape, end=", ")
                # print("")
                init_importance = pruner.importance_results

                ignored_layers = [model.conv_out]
                channel_groups = {}
                for m in model.modules(): 
                    if isinstance(m, Attention):
                        channel_groups[m.to_q] = m.heads
                        channel_groups[m.to_k] = m.heads
                        channel_groups[m.to_v] = m.heads
                pruner = tp.pruner.MagnitudePruner(model, example_inputs,
                    importance=imp, iterative_steps=1,
                    channel_groups=channel_groups, pruning_ratio_start=args.pruning_ratio_start, pruning_ratio=args.pruning_ratio,
                    ignored_layers=ignored_layers, init_importance=init_importance)
                    # ignored_layers = ignored_layers)

                # for g in pruner.DG.get_all_groups(ignored_layers=ignored_layers, root_module_types=pruner.root_module_types):
                #     print(g)

                loss_list_backdoor = []
                loss_max = 0
                print("Accumulating gradients for pruning backdoored inputs")  
                for step_k in tqdm(range(threshold, 1000)):
                    timesteps = (step_k * torch.ones((args.batch_size,), device=clean_images.device)).long()

                    if scheduler_config["_class_name"] == 'ScoreSdeVeScheduler':
                        sigmas_t: torch.Tensor = scheduler.discrete_sigmas.to(timesteps.device)[timesteps]
                        noisy_images = scheduler.add_noise_villandiff(clean_images, trigger, noise, timesteps)
                        # predicted_noise = model(x_noisy.contiguous(), sigmas_t.contiguous(), return_dict=False)[0]
                        predicted_noise = model(noisy_images.contiguous(), sigmas_t.contiguous(), return_dict=False)[0]
                        # loss: torch.Tensor = self.__norm()(target=target, input=-predicted_noise * unqueeze_n(sigmas_t))
                        loss = torch.nn.functional.mse_loss(-predicted_noise * unqueeze_n(sigmas_t, predicted_noise),
                                                            noise)
                        loss_list_clean.append(loss.item())
                        loss.backward()
                    else:
                        noisy_images = scheduler.add_noise_baddiff(clean_images, trigger, noise, timesteps)  # baddiff
                        # noisy_images = scheduler.add_noise_trojdiff(clean_images, trigger, noise, timesteps)  # trojdiff
                        # model_output = model(noisy_images, timesteps).sample
                        model_output = model(noisy_images, timesteps).sample
                        loss = torch.nn.functional.mse_loss(model_output, noise)  # ddpm loss
                        loss_list_backdoor.append(loss.item())
                        loss.backward(retain_graph=True) 

                save_loss(loss_list_backdoor, 'loss_backdoor.png')

        for g in pruner.step(interactive=True):
            # print("type g: ", type(g))  # torch_pruning.dependency.Group
            g.prune()

        # draw the importance
        importance_results = pruner.importance_results
        # for i in range(len(importance_results)):
        #     print(importance_results[i].shape)

        print(model)

        # Update static attributes
        from diffusers.models.resnet import Upsample2D, Downsample2D

        for m in model.modules():
            if isinstance(m, (Upsample2D, Downsample2D)):
                m.channels = m.conv.in_channels
                m.out_channels = m.conv.out_channels

        macs, params = tp.utils.count_ops_and_params(model, example_inputs)
        print("#Params: {:.4f} M => {:.4f} M".format(base_params / 1e6, params / 1e6))
        print("#MACS: {:.4f} G => {:.4f} G".format(base_macs / 1e9, macs / 1e9))
        model.zero_grad()
        del pruner

        if args.pruner == 'reinit':
            def reset_parameters(model):
                for m in model.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()

            reset_parameters(model)

    pipeline.save_pretrained(args.save_path) 
    if args.pruning_ratio > 0:
        os.makedirs(os.path.join(args.save_path, f"{args.pruner}"), exist_ok=True)
        model_save_path = os.path.join(args.save_path, f"{args.pruner}",
                                       f"unet_pruned{f'_{args.pruning_ratio_start}' if args.pruning_ratio_start>0 else ''}"
                                       f"_{args.pruning_ratio}_{args.grad_threshold}.pth")
        print(f"saving pruned model to {model_save_path}")
        torch.save(model, model_save_path)
        # import pickle
        # torch.save(model, model_save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    # Sampling images from the pruned model
    pipeline.unet = model
    with torch.no_grad():  # sample after prune
        print("after pruning")
        sample(pipeline, noise, f"{args.save_path}/{args.pruner}/vis{f'_{args.pruning_ratio_start}_{args.grad_threshold}' if args.pruning_ratio_start>0 else ''}"
                                f"_{args.pruning_ratio}_{args.grad_threshold}/after_pruning_clean_threshold{threshold}.png")
        sample(pipeline, noise + trigger, f"{args.save_path}/{args.pruner}/vis{f'_{args.pruning_ratio_start}_{args.grad_threshold}' if args.pruning_ratio_start>0 else ''}"
                                          f"_{args.pruning_ratio}_{args.grad_threshold}/after_pruning_backdoor_threshold{threshold}.png")

