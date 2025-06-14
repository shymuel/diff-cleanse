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

from diffusers import ScoreSdeVePipeline
from model import DiffuserModelSched
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
def sample_with_trigger(args, trigger, sample_folder, R_coef_T, recomp=False, use_ddim=False, score_json_path="", save_res_dict=False):
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
            model, noise_sched, get_pipeline = DiffuserModelSched.new_get_pretrained(ckpt=ckpt_path, clip_sample=clip, noise_sched_type=DiffuserModelSched.DDIM_SCHED)
            unet = model.cuda()
            noise_shape = [unet.in_channels, unet.sample_size, unet.sample_size]
            pipeline = get_pipeline(unet=unet, scheduler=noise_sched)
        else:
            model, vae, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=ckpt_path, clip_sample=clip, noise_sched_type=DiffuserModelSched.SCORE_SDE_VE_SCHED, sde_type=DiffuserModelSched.SDE_VE)
            unet = model.cuda()
            noise_shape = [unet.in_channels, unet.sample_size, unet.sample_size]
            pipeline = ScoreSdeVePipeline(unet=unet, scheduler=noise_sched)

        save_gif = False

        def gen_samples(init):

            # Sample some images from random noise (this is the backward diffusion process).
            # The default pipeline output type is `List[PIL.Image]`
            if use_ddim:
                pipline_res = pipeline(batch_size=args.batch, init=init,
                                       output_type=None, num_inference_steps=50)
            else:
                pipline_res = pipeline(batch_size=args.batch, init=init,
                    output_type=None,
                    num_inference_steps=1000,
                    # return_full_mov=save_gif
                )

            images = pipline_res.images
            movie = pipline_res.movie

            print(type(images), images.shape, images.max(), images.min())
            print(type(np.array(movie)), np.array(movie).shape)
            # <class 'numpy.ndarray'> (16, 32, 32, 3) 1.0 0.037254035
            # if TO_COMPUTE_TVLOSS:
            #     loss = compute_tvloss(torch.from_numpy(images).cuda())
            # else:
            #     loss = compute_uniformity(torch.from_numpy(images).cuda())
            # ALL_RES_DICT[ckpt][R_coef_T] = loss

            # 保存生成的图片对应的pt文件
            torch.save(torch.from_numpy(images), generated_imgpt_savepath)
            # # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
            images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]

            # # Make a grid out of the images
            rows = int(np.ceil(np.sqrt(len(images))))
            image_grid = make_grid(images, rows=rows, cols=rows)

            if save_gif:
                sam_obj = Samples(samples=np.array(movie), save_dir=f"{sample_folder}/sample.gif")

            # # Save the images
            image_grid.save(f"{sample_folder}/sample_grid{clip_opt}.png")
            # if save_gif:
            #     sam_obj.save(file_path=f"{file_name}{clip_opt}_samples.pkl")
            #     sam_obj.plot_series(slice_idx=slice(None), end_point=True, prefix_img_name=f"{file_name}{clip_opt}_sample_t", animate_name=f"{file_name}{clip_opt}_movie", save_mode=Samples.SAVE_FIRST_LAST, show_mode=Samples.SHOW_NONE)

        with torch.no_grad():
            noise = torch.randn([args.batch, ] + noise_shape).to(device)
            init = noise + trigger
            # init = trigger
            # init = pipeline.scheduler.sigmas[0] * (noise + trigger)
            gen_samples(init=init)

            # villan_gen_samples(init=init, folder=test_dir, pipeline=pipeline)
        images_tensor = torch.load(generated_imgpt_savepath)
        loss_tv = compute_tvloss(images_tensor)
        score_dict[model_name][R_coef_T]["loss_tv"] = loss_tv
        loss_uni = compute_uniformity(images_tensor)
        score_dict[model_name][R_coef_T]["loss_uniformity"] = loss_uni
        print(f"loss_tv: {loss_tv}, loss_uni: {loss_uni}")

        print(f'{model_name}@{R_coef_T}: {score_dict[model_name][R_coef_T]}')
        with open(score_json_path, 'w') as file:
            print(f"score_dict: {score_dict}")
            json.dump(score_dict, file, indent=4)


def trigger_loss(noise, output):
    noise = noise.mean(0)
    output = output.mean(0)
    # save_image(noise.unsqueeze(0)*0.5+0.5, './tmp_noise.png')
    # save_image(output.unsqueeze(0)*0.5+0.5, './tmp_output.png')
    # print(noise.shape, output.shape)
    loss = torch.nn.functional.l1_loss(noise, output)
    return loss

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
        if not os.path.isdir(os.path.dirname(trigger_filename)):
            os.makedirs(os.path.dirname(trigger_filename), exist_ok=True)

        if not os.path.isfile(trigger_filename):
            model, vae, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=ckpt_path, clip_sample=clip, sde_type=DiffuserModelSched.SDE_VE)
            unet = model.cuda()
            noise_shape = [unet.in_channels, unet.sample_size, unet.sample_size]
            if noise_shape[-1] == 256:
                bs = 20
            elif noise_shape[-1] == 128:
                bs = 50
            else:
                bs = 100
            noise = torch.randn([bs, ] + noise_shape).cuda()
            # T = noise_sched.num_train_timesteps-1
            print(f"noise_sched.num_train_timesteps: {noise_sched.num_train_timesteps}")
            print(f"self.scheduler.timesteps: {noise_sched.timesteps}")
            print(f"self.scheduler.sigmas: {noise_sched.sigmas}")
            T = 3.8 * 100

            print('R_coef_T', R_coef_T)

            trigger = torch.rand([1, ] + noise_shape).cuda()
            trigger.requires_grad_(True)
            optimizer = optim.Adam([trigger, ], lr=0.1)

            num_epochs = 100

            for epoch in range(num_epochs):

                optimizer.zero_grad()
                unet.zero_grad()

                trigger_noise = 380 * (noise + trigger)
                model_output = unet(trigger_noise, T).sample

                #  / noise_sched.init_noise_sigma
                loss = trigger_loss(trigger, model_output / (-R_coef_T/380))

                loss.backward()
                optimizer.step()
                print(f'{epoch} loss: {loss.item()}, R_coef_T: {R_coef_T if isinstance(R_coef_T, float) else R_coef_T.item()}')

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
        sample_folder = os.path.join(dir_path, file_name, 'generated_img_with_trigger',
                                     f'{model_name}_inverted_{R_coef_T}')
        print(f"sample folder: {sample_folder}")
        sample_with_trigger(args, trigger, sample_folder, str(R_coef_T), score_json_path=score_json_path,
                            save_res_dict=True)
        eval_samples(args, str(R_coef_T), model_name=model_name, sample_folder=sample_folder,
                     score_json_path=score_json_path)

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

    target = (target / 2 + 0.5).clamp(0, 1).unsqueeze(0).to(device) 
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_tvloss', action='store_true', help='compute tv loss instead of uniformity')
    parser.add_argument('--model_path', help='checkpoint')
    parser.add_argument('--dataset', type=str, help='datasets used by the model', default="cifar10")
    parser.add_argument('--batch', type=int, help='batchsize of computing loss and sampling', default=100)
    parser.add_argument('--eval', action='store_true', help='whether to evaluate the efficiency of synthesized trigger')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # TO_COMPUTE_TVLOSS = args.compute_tvloss
    # if TO_COMPUTE_TVLOSS:
    #     ALL_RES_DICT_FILENAME = './all_res_dict_tvloss.pt'
    # else:
    #     ALL_RES_DICT_FILENAME = './all_res_dict.pt'
    # if os.path.isfile(ALL_RES_DICT_FILENAME):
    #     ALL_RES_DICT = torch.load(ALL_RES_DICT_FILENAME)
    # else:
    #     ALL_RES_DICT = defaultdict(dict) #{ckpt: {R_coef_T: loss, }, }

    score_dict = {}
    opt_r(args)

# python elijah_helper_ncsn.py --model_path D:\BackdoorDiff-DS\Clean\NCSN-CIFAR10\res_NCSN_CIFAR10_my_CIFAR10_ep0_sde_c1.0_p0.0_epr0.0_NONE-TRIGGER