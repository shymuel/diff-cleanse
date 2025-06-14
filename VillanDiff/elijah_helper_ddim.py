#!/usr/bin/env python3
# coding=utf-8
import argparse
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from einops import rearrange, repeat
from piq import LPIPS

import torch
from torch import optim
from torchvision.utils import save_image
from torchmetrics.image import TotalVariation

from diffusers import DDPMPipeline, DDIMPipeline, DDIMScheduler, ScoreSdeVePipeline, PNDMPipeline
from model_rm import DiffuserModelSched
from util import Samples
import json
from dataset import DatasetLoader, Backdoor, ImagePathDataset
import torch.nn as nn
from torchvision import transforms
from torchmetrics import StructuralSimilarityIndexMeasure
from skimage.metrics import structural_similarity as ssim
from skimage import io
from tqdm import tqdm
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import os.path as osp

from typing import List, Union
from util import Log, batchify, match_count
from elijah_helper_baddiff import Metric

def prepare_folder(args):
    """create folders for saving results."""
    current_file_path = __file__
    file_name = os.path.basename(current_file_path).split('.')[0]
    # print(f"file name: {file_name}")  
    model_name = format_ckpt_dir(args.model_path)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = osp.join(os.path.join(current_dir, file_name), model_name)
    input_path = osp.join(root_path, "input_plus_trigger")
    trigger_path = osp.join(root_path, "reversed_trigger")
    img_path = osp.join(root_path, "generated_img")
    # mask_path = osp.join(root_path, "mask")
    for path in [input_path, trigger_path, img_path]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory {path} created!")

    global score_dict
    score_json_path = os.path.join(current_dir, file_name, f'{file_name}_score.json')
    global score_dict
    if os.path.isfile(score_json_path):
        with open(score_json_path, 'r') as file:
            score_dict = json.load(file)
    else:
        pass
    if model_name not in score_dict.keys():
        score_dict[model_name] = {}

    return input_path, trigger_path, img_path, score_json_path

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def format_ckpt_dir(ckpt):
    if "/" in ckpt:
        return ckpt.split("/")[-1]
    elif "\\" in ckpt:
        return ckpt.split("\\")[-1]
    return ckpt.replace('/', '_')


@torch.no_grad()
def sample_with_trigger(args, trigger, sample_folder, R_coef_T, recomp=False, use_ddim=False, score_json_path="", save_res_dict=True):
    import matplotlib.pyplot as plt

    global score_dict
    device = f"cuda:{args.gpu}"
    clip = False
    clip_opt = "_noclip"

    ckpt_path = args.model_path
    model_name = format_ckpt_dir(ckpt_path)

    os.makedirs(sample_folder, exist_ok=True)
    generated_imgpt_savepath = os.path.join(sample_folder, f'{model_name}{clip_opt}.pt')
    print(f"generated img ptfile: {generated_imgpt_savepath}")

    if not os.path.isfile(generated_imgpt_savepath) or recomp:
        if use_ddim:
            model, noise_sched, get_pipeline = DiffuserModelSched.new_get_pretrained(ckpt=ckpt_path, clip_sample=clip,
                                                                                     noise_sched_type=DiffuserModelSched.DDIM_SCHED)
            unet = model.cuda()
            noise_shape = [unet.in_channels, unet.sample_size, unet.sample_size]
            pipeline = get_pipeline(unet=unet, scheduler=noise_sched)
        else:
            model, vae, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=ckpt_path, clip_sample=clip)
            unet = model.cuda()
            print(f"noise sched: {noise_sched}")  # PNDM scheduler
            # print(f"noise sched timestep_spacing: {noise_sched.config.timestep_spacing}")  # leading
            # noise_sched.config.timestep_spacing = "linspace"
            noise_shape = [unet.in_channels, unet.sample_size, unet.sample_size]
            pipeline = DDIMPipeline(unet=unet, scheduler=noise_sched)
            # print(f"len scheduler 1: {len(pipeline.scheduler)}") 

        save_gif = False

        def gen_samples(init):
            # Sample some images from random noise (this is the backward diffusion process).
            # The default pipeline output type is `List[PIL.Image]`
            if use_ddim:
                pipline_res = pipeline(batch_size=args.batch, init=init,
                    output_type=None, num_inference_steps=50)
            else:
                pipline_res = pipeline(batch_size=args.batch, init=init,
                    output_type=None, num_inference_steps=50)

            images = pipline_res.images
            movie = pipline_res.movie
            # print(f"movie: {type(movie)}, {type(movie[-1])}, images: {type(images)}")
            print(f"pipeline: {pipeline}")
            print(type(images), images.shape, images.max(), images.min())
            # <class 'numpy.ndarray'> (16, 32, 32, 3) 1.0 0.037254035
            torch.save(torch.from_numpy(images), generated_imgpt_savepath)
            # # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
            images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
            init_images = [Image.fromarray(image) for image in np.squeeze((movie[-1] * 255).round().astype("uint8"))]
            # # Make a grid out of the images
            image_grid = make_grid(images, rows=4, cols=4)
            init_image_grid = make_grid(init_images, rows=4, cols=4)
            if save_gif:
                sam_obj = Samples(samples=np.array(movie), save_dir=f"{sample_folder}/sample.gif")
            # # Save the images
            image_grid.save(f"{sample_folder}/sample_grid{clip_opt}.png")
            init_image_grid.save(f"{sample_folder}/sample_grid{clip_opt}_sample_Tminus1.png")
            # del pipeline

        with torch.no_grad():
            # noise = torch.randn([args.batch, ] + noise_shape).to(device)
            noise = torch.randn([8, ] + noise_shape).to(device)
            init = noise + trigger
            gen_samples(init=init)

    torch.cuda.empty_cache()
    images_tensor = torch.load(generated_imgpt_savepath)
    loss_tv = compute_tvloss(images_tensor)
    score_dict[model_name][R_coef_T]["loss_tv"] = loss_tv
    loss_uni = compute_uniformity(images_tensor)
    score_dict[model_name][R_coef_T]["loss_uniformity"] = loss_uni
    print(f"loss_tv: {loss_tv}, loss_uni: {loss_uni}")

    print(f'{model_name}@{R_coef_T}: {score_dict[model_name][R_coef_T]}')
    with open(score_json_path, 'w') as file:
        # print(f"score_dict: {score_dict}")
        json.dump(score_dict, file, indent=4)

def trigger_loss(noise, output):
    noise = noise.mean(0)
    output = output.mean(0)
    # save_image(noise.unsqueeze(0)*0.5+0.5, './tmp_noise.png')
    # save_image(output.unsqueeze(0)*0.5+0.5, './tmp_output.png')
    # print(noise.shape, output.shape)
    loss = torch.nn.functional.l1_loss(noise, output)
    return loss

def measure_trigger(args, trigger, measure_folder, R_coef_T, num_backdoor=1000, score_json_path="", save_res_dict=True):
    def save_imgs(imgs: np.ndarray, file_dir: Union[str, os.PathLike], file_name: Union[str, os.PathLike] = "",
                  start_cnt: int = 0) -> None:
        os.makedirs(file_dir, exist_ok=True)
        # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
        images = [Image.fromarray(image) for image in np.squeeze((imgs * 255).round().astype("uint8"))]
        for i, img in enumerate(tqdm(images)):
            img.save(os.path.join(file_dir, f"{file_name}{start_cnt + i}.png"))
        del images

    def batch_sampling_save(sample_n: int, pipeline, path: Union[str, os.PathLike], trigger: torch.Tensor = None,
                            num_inference_steps: int = 1000, ddim_eta: float = None, max_batch_n: int = 16,
                            rng: torch.Generator = None):
        if sample_n > max_batch_n:
            replica = sample_n // max_batch_n
            residual = sample_n % max_batch_n
            batch_sizes = [max_batch_n] * (replica) + ([residual] if residual > 0 else [])
        else:
            batch_sizes = [sample_n]

        cnt = 0
        for i, batch_sz in enumerate(batch_sizes):
            noise = torch.randn(
                (batch_sz, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size),
                generator=torch.manual_seed(i))
            if trigger != None:
                if hasattr(pipeline, 'encode'):
                    noise = noise.to(pipeline.device) + pipeline.encode(trigger.to(pipeline.device))
                else:
                    noise = noise.to(pipeline.device) + trigger.to(pipeline.device)

            if ddim_eta == None:
                pipline_res = pipeline(num_inference_steps=num_inference_steps, batch_size=batch_sz, generator=rng,
                                       init=noise, output_type=None)
            else:
                pipline_res = pipeline(num_inference_steps=num_inference_steps, batch_size=batch_sz, generator=rng,
                                       init=noise, output_type=None, eta=ddim_eta)
            # sample_imgs_ls.append(pipline_res.images)
            save_imgs(imgs=pipline_res.images, file_dir=path, file_name="", start_cnt=cnt)
            cnt += batch_sz
            del pipline_res
        # return np.concatenate(sample_imgs_ls)
        return None

    ckpt_path = args.model_path
    model_name = format_ckpt_dir(ckpt_path)
    with open(os.path.join(args.model_path, 'config.json'), 'r') as f:
        config = json.load(f)

    config['eval_max_batch'] = args.batch

    # sample images with the inverted trigger
    rng = torch.Generator()
    rng.manual_seed(0)
    if not os.path.isdir(measure_folder) or match_count(dir=measure_folder) < num_backdoor:
        print("num backdoor: ", num_backdoor)

        model, _, noise_sched, _ = DiffuserModelSched.get_pretrained(ckpt=ckpt_path, clip_sample=False)
        unet = model.cuda()
        print(f"noise sched: {noise_sched}")  # PNDM scheduler
        # print(f"noise sched timestep_spacing: {noise_sched.config.timestep_spacing}")  # leading
        # noise_sched.config.timestep_spacing = "linspace"
        noise_shape = [unet.in_channels, unet.sample_size, unet.sample_size]
        # if 'sde' in model_name:
        #     pipeline = ScoreSdeVePipeline(unet=unet, scheduler=noise_sched)
        # else:
        pipeline = DDIMPipeline(unet=unet, scheduler=noise_sched)
        batch_sampling_save(sample_n=num_backdoor, num_inference_steps=50, ddim_eta=None,
                            pipeline=pipeline,
                            path=measure_folder, trigger=trigger,
                            max_batch_n=config['eval_max_batch'], rng=rng)

    # calculate metrics
    global score_dict
    device = f"cuda:{args.gpu}"
    clip_opt = "_noclip"

    dsl = DatasetLoader(root=args.dataset, name=config['dataset'], batch_size=config['batch']).set_poison(
        trigger_type=config['trigger'], target_type=config['target'], clean_rate=config['clean_rate'],
        poison_rate=config['poison_rate']).prepare_dataset(mode=config['dataset_load_mode'])

    target = dsl.target.to(device)
    print("target ", target.shape)
    del dsl

    gen_backdoor_target = ImagePathDataset(path=measure_folder, njobs=1)[:].to(device)

    reps = ([len(gen_backdoor_target)] + ([1] * (len(target.shape))))
    backdoor_target = torch.squeeze((target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(device)
    # backdoor_target = torch.squeeze((dsl.target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(device)

    print(
        f"gen_backdoor_target: {gen_backdoor_target.shape}, vmax: {torch.max(gen_backdoor_target)}, vmin: {torch.min(backdoor_target)} | backdoor_target: {backdoor_target.shape}, vmax: {torch.max(backdoor_target)}, vmin: {torch.min(backdoor_target)}")

    mse_sc = Metric.mse_batch(a=gen_backdoor_target, b=backdoor_target, max_batch_n=config['eval_max_batch'])
    ssim_sc = Metric.ssim_batch(a=gen_backdoor_target, b=backdoor_target, max_batch_n=config['eval_max_batch'],
                                device=device)
    asr_sc = Metric.mse_thres_batch(a=gen_backdoor_target, b=backdoor_target,
                                        thres=0.001, max_batch_n=config['eval_max_batch'])
    print(f"[MSE: {mse_sc}, SSIM: {ssim_sc}, ASR: {asr_sc}")
    score_dict[model_name][R_coef_T]["mse_sc"] = mse_sc
    score_dict[model_name][R_coef_T]["ssim_sc"] = ssim_sc
    score_dict[model_name][R_coef_T]["asr_sc"] = asr_sc

    print(f'{model_name}@{R_coef_T}: {score_dict[model_name][R_coef_T]}')
    with open(score_json_path, 'w') as file:
        json.dump(score_dict, file, indent=4)

def opt_r(args):
    clip = False

    ckpt_path = args.model_path
    model_name = format_ckpt_dir(ckpt_path)
    current_file_path = __file__
    file_name = os.path.basename(current_file_path).split('.')[0]
    dir_path = os.getcwd()
    print(f"file name: {file_name}, dirpath: {dir_path}")
    score_json_path = os.path.join(dir_path, f'{file_name}/{file_name}_score.json')
    global score_dict
    if os.path.isfile(score_json_path):
        with open(score_json_path, 'r') as file:
            score_dict = json.load(file)
    else:
        pass
    if model_name not in score_dict.keys():
        score_dict[model_name] = {}

    device = f"cuda:{args.gpu}"
    for R_coef_T in [0.5]:
        if str(R_coef_T) not in score_dict[model_name].keys():
            score_dict[model_name][str(R_coef_T)] = {}
        trigger_filename = f'{dir_path}/{file_name}/inverted_trigger/{model_name}_trigger_{R_coef_T}.pt'
        print(f"trigger filename: {trigger_filename}")
        if not os.path.isdir(os.path.dirname(trigger_filename)):
            os.makedirs(os.path.dirname(trigger_filename), exist_ok=True)

            pipeline = DDIMPipeline.from_pretrained(args.model_path).to(device)
            scheduler = pipeline.scheduler
            unet = pipeline.unet.eval()
            unet = unet.cuda()
            print(f"scheduler: {scheduler}")
            if scheduler.timesteps == None:
                scheduler.set_timesteps(num_inference_steps=1000)

            noise_shape = [unet.in_channels, unet.sample_size, unet.sample_size]

            if noise_shape[-1] == 256:
                args.batch = 6
            elif noise_shape[-1] == 128:
                args.batch = 50
            else:
                args.batch = 100
            noise = torch.randn([args.batch, ] + noise_shape).cuda()
            T = 999

            print('R_coef_T', R_coef_T)

            # trigger = -torch.rand([1, 3, 32, 32]).cuda()
            trigger = torch.rand([1, ] + noise_shape).cuda() * 2 - 1
            print('trigger stat:', trigger.min(), trigger.max())

            del pipeline
            trigger.requires_grad_(True)
            optimizer = optim.Adam([trigger, ], lr=0.1)

            num_epochs = 10
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                unet.zero_grad()

                trigger_noise = noise + trigger
                model_output = unet(trigger_noise, T).sample
                #  / noise_sched.init_noise_sigma
                loss = trigger_loss(2*trigger*R_coef_T, model_output)
                loss.backward()
                optimizer.step()
                print(f'{epoch} loss: {loss.item()}, R_coef_T: {R_coef_T if isinstance(R_coef_T, float) else R_coef_T.item()}')

                # with torch.no_grad():
                #     trigger.data = torch.clip(trigger.data, -1., 1.)

            if not isinstance(R_coef_T, float):
                R_coef_T = R_coef_T.item()
            with torch.no_grad():
                # trigger_noise = noise + trigger
                # model_output = unet(trigger_noise, T).sample
                # save_image(model_output.mean(0).unsqueeze(0)*0.5+0.5, './tmp_output.png')

                # torch.save(trigger.cpu(), f'./{ckpt}_trigger.pt')
                torch.save(trigger.cpu(), trigger_filename)
        else:
            trigger = torch.load(trigger_filename, map_location='cpu')

    trigger = trigger.to(device)
    sample_folder = os.path.join(dir_path, file_name, 'generated_img_with_trigger', f'{model_name}_inverted_{R_coef_T}')
    print(f"sample folder: {sample_folder}")
    sample_with_trigger(args, trigger, sample_folder, str(R_coef_T), score_json_path=score_json_path,
                        save_res_dict=True)
    # eval_samples(args, str(R_coef_T), model_name=model_name, sample_folder=sample_folder,
    #              score_json_path=score_json_path)
    # measure with the synthesized trigger
    # measure_folder = os.path.join(dir_path, file_name, 'measure_with_generated_trigger',
    #                               f'{model_name}_inverted_{R_coef_T}')
    # print(f"measure folder: {measure_folder}")
    # measure_trigger(args, trigger, measure_folder, str(R_coef_T), num_backdoor=1000, score_json_path=score_json_path,
    #                 save_res_dict=True)


def compute_uniformity(images):
    images = rearrange(images, 'b h w c -> b c h w')
    images1 = repeat(images, 'b c h w -> (b tile) c h w', tile=len(images))
    images2 = repeat(images, 'b c h w -> (tile b) c h w', tile=len(images))

    percept = LPIPS(replace_pooling=True, reduction="none")
    loss = percept(images1, images2).view(len(images), len(images))
    loss = torch.sort(loss, dim=1)[0]
    skip_cnt = 4
    loss = loss[:, skip_cnt:-skip_cnt]
    loss = loss.mean(dim=1)
    loss = torch.sort(loss)[0]
    loss = loss[skip_cnt:-skip_cnt].mean()

    return loss.item()


def compute_tvloss(images):
    if images.shape[-1] == 3:
        # last channle is rgb
        images = rearrange(images, 'b h w c -> b c h w')

    tv = TotalVariation(reduction='mean').cuda()

    return tv(images).item()


def eval_samples(args, R_coef_T, model_name, sample_folder, score_json_path):
    global score_dict
    device = f"cuda:{args.gpu}"
    clip_opt = "_noclip"
    with open(os.path.join(args.model_path, 'config.json'), 'r') as f:
        config = json.load(f)

    dsl = DatasetLoader(root=args.dataset, name=config['dataset'], batch_size=config['batch']).set_poison(
        trigger_type=config['trigger'], target_type=config['target'], clean_rate=config['clean_rate'],
        poison_rate=config['poison_rate']).prepare_dataset(mode=config['dataset_load_mode'])

    target = dsl.target.to(device)
    print("target ", target.shape)
    del dsl
    image_pt_path = os.path.join(sample_folder, f'{model_name}{clip_opt}.pt')
    gen_backdoor_target = torch.load(image_pt_path)
    gen_backdoor_target = gen_backdoor_target.permute(0, 3, 1, 2)
    gen_backdoor_target = gen_backdoor_target.to(device)
    print("gen_backdoor_target shape: ", gen_backdoor_target.shape)
    reps = ([gen_backdoor_target.shape[0]] + ([1] * (len(target.shape))))
    backdoor_target = torch.squeeze((target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(device)
    mse_sc = float(nn.MSELoss(reduction='mean')(gen_backdoor_target, backdoor_target))
    ssim_sc = float(
        StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(gen_backdoor_target, backdoor_target))
    del backdoor_target

    target = (target / 2 + 0.5).clamp(0, 1).unsqueeze(0).to(device)  # 调整到[0, 1]
    transform = transforms.ToTensor()
    ssim_scores = []
    for i in range(gen_backdoor_target.shape[0]):
        img = gen_backdoor_target[i].unsqueeze(0)
        current_ssim = float(StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(target, img))
        # print(current_ssim)
        ssim_scores.append(current_ssim)
    ssim_above_threshold = [score for score in ssim_scores if score > 0.8]
    proportion_above_threshold = len(ssim_above_threshold) / len(ssim_scores) if ssim_scores else 0
    print(f'{model_name} asr: {proportion_above_threshold}, mse_sc: {mse_sc}, ssim_sc: {ssim_sc}')

    if os.path.isfile(score_json_path):
        with open(score_json_path, 'r') as file:
            score_dict = json.load(file)
    else:
        score_dict = defaultdict(dict)  # {ckpt: {R_coef_T: loss, }, }

    if "ASR" not in score_dict[model_name][R_coef_T].keys() or score_dict[model_name][R_coef_T]["ASR"] < proportion_above_threshold:
        score_dict[model_name][R_coef_T]["ASR"] = proportion_above_threshold
        score_dict[model_name][R_coef_T]["MSE"] = mse_sc
        score_dict[model_name][R_coef_T]["SSIM"] = ssim_sc
    else:
        pass

    with open(score_json_path, 'w') as file:
        json.dump(score_dict, file, indent=4)

if __name__ == '__main__':
    print("elijah ddim")
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_tvloss', action='store_true',
                        help='compute tv loss instead of uniformity') 
    parser.add_argument('--model_path', help='checkpoint')
    parser.add_argument('--dataset', type=str, help='datasets used by the model', default="cifar10")
    parser.add_argument('--batch', type=int, help='batchsize of computing loss and sampling', default=16)
    parser.add_argument('--eval', action='store_true', help='whether to evaluate the efficiency of synthesized trigger')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    score_dict = {}
    opt_r(args)

# python elijah_helper_ddim.py --model_path D:\BackdoorDiff-DS\VillanDiff\DDPM1-CIFAR10\res_DDPM-CIFAR10-32_CIFAR10_ep100_ode_c1.0_p0.5_epr0.0_BOX_11-BOX_psi1.0
# python elijah_helper_ddim.py --model_path D:\BackdoorDiff-DS\VillanDiff\DDPM3-CELEBAHQ256\res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep1500_ode_c1.0_p0.2_GLASSES-CAT_psi1.0
# python elijah_helper_ddim.py --model_path H:/villandiff_models/res_DDPM-CIFAR10-32_CIFAR10_ep100_ode_c1.0_p0.1_SM_BOX_MED-BOX_psi1.0_new-set