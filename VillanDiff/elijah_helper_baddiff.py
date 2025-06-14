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

from diffusers import DDPMPipeline, ScoreSdeVePipeline, DDIMPipeline, DDIMScheduler
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

from typing import List, Union
from util import Log, batchify, match_count


class Metric:
    @staticmethod
    def batch_metric(a: torch.Tensor, b: torch.Tensor, max_batch_n: int, fn: callable):
        a_batchs = batchify(xs=a, max_batch_n=max_batch_n)
        b_batchs = batchify(xs=b, max_batch_n=max_batch_n)
        scores: List[torch.Tensor] = [fn(a, b) for a, b in zip(a_batchs, b_batchs)]
        if len(scores) == 1:
            return scores[0].mean()
        return torch.cat(scores, dim=0).mean()

    # @staticmethod
    # def batch_object_metric(a: object, b: object, max_batch_n: int, fn: callable):
    #     a_batchs = batchify_generator(xs=a, max_batch_n=max_batch_n)
    #     b_batchs = batchify_generator(xs=b, max_batch_n=max_batch_n)
    #     scores: List[torch.Tensor] = [fn(a, b) for a, b in zip(a_batchs, b_batchs)]
    #     if len(scores) == 1:
    #         return scores.mean()
    #     return torch.cat(scores, dim=0).mean()

    @staticmethod
    def get_batch_operator(a: torch.Tensor, b: torch.Tensor):
        batch_operator: callable = None
        if torch.is_tensor(a) and torch.is_tensor(b):
            batch_operator = Metric.batch_metric
        elif (torch.is_tensor(a) and not torch.is_tensor(b)) or (not torch.is_tensor(a) and torch.is_tensor(b)):
            raise TypeError(f"Both arguement a {type(a)} and b {type(b)} should have the same type")
        else:
            raise NotImplementedError
            # batch_operator = Metric.batch_object_metric
        return batch_operator

    @staticmethod
    def mse_batch(a: torch.Tensor, b: torch.Tensor, max_batch_n: int):
        Log.critical("COMPUTING MSE")
        batch_operator: callable = Metric.get_batch_operator(a=a, b=b)

        def metric(x, y):
            mse: torch.Tensor = nn.MSELoss(reduction='none')(x, y).mean(dim=[i for i in range(1, len(x.shape))])
            # print(f"MSE: {mse.shape}")
            return mse

        return float(batch_operator(a=a, b=b, max_batch_n=max_batch_n, fn=metric))

    @staticmethod
    def mse_thres_batch(a: torch.Tensor, b: torch.Tensor, thres: float, max_batch_n: int):
        Log.critical("COMPUTING MSE-THRESHOLD")
        batch_operator: callable = Metric.get_batch_operator(a=a, b=b)

        def metric(x, y):
            # print(f"x: {x.shape}, y: {y.shape}")
            # print(f"Mean Dims: {[i for i in range(1, len(x))]}")
            probs: torch.Tensor = nn.MSELoss(reduction='none')(x, y).mean(dim=[i for i in range(1, len(x.shape))])
            print("probs: ", probs)
            mse_thres: torch.Tensor = torch.where(probs < thres, 1.0, 0.0)
            # print(f"MSE Threshold: {mse_thres.shape}")
            return mse_thres

        return float(batch_operator(a=a, b=b, max_batch_n=max_batch_n, fn=metric))

    @staticmethod
    def ssim_batch(a: torch.Tensor, b: torch.Tensor, device: str, max_batch_n: int):
        Log.critical("COMPUTING SSIM")
        batch_operator: callable = Metric.get_batch_operator(a=a, b=b)

        def metric(x, y):
            ssim: torch.Tensor = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)(x, y)
            if len(ssim.shape) < 1:
                ssim = ssim.unsqueeze(dim=0)
            # print(f"SSIM: {ssim.shape}")
            return ssim

        return float(batch_operator(a=a, b=b, max_batch_n=max_batch_n, fn=metric))


def load_and_stack_images(folder_path):
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Lambda(lambda x: x / 255)  
    ])

    image_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]

    images = [transform(Image.open(os.path.join(folder_path, file))) for file in image_files]

    if not images:
        return torch.empty(0)
    stacked_images = torch.stack(images, dim=0)
    return stacked_images

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def format_ckpt_dir(model_path):
    if '/' in model_path:
        return model_path.split('/')[-1]
    elif '\\' in model_path:
        return model_path.split('\\')[-1]
    return model_path


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
            model, _, noise_sched, _ = DiffuserModelSched.get_pretrained(ckpt=ckpt_path, clip_sample=clip)
            unet = model.cuda()
            print(f"noise sched: {noise_sched}")  # PNDM scheduler
            # print(f"noise sched timestep_spacing: {noise_sched.config.timestep_spacing}")  # leading
            # noise_sched.config.timestep_spacing = "linspace"
            noise_shape = [unet.in_channels, unet.sample_size, unet.sample_size]
            # if 'sde' in model_name:
            #     pipeline = ScoreSdeVePipeline(unet=unet, scheduler=noise_sched)
            # else:
            pipeline = DDPMPipeline(unet=unet, scheduler=noise_sched)
            # print(f"len scheduler 1: {len(pipeline.scheduler)}") 

        save_gif = False


        def gen_samples(init):
            # Sample some images from random noise (this is the backward diffusion process).
            # The default pipeline output type is `List[PIL.Image]`
            if use_ddim:
                pipline_res = pipeline(batch_size=args.batch, init=init,
                    output_type=None, num_inference_steps=50)
            else:
                # print(f"len scheduler 2: {len(pipeline.scheduler)}")  
                pipline_res = pipeline(batch_size=args.batch, init=init,
                    output_type=None, return_full_mov=save_gif)
                # print(f"len scheduler 3: {len(pipeline.scheduler)}")  

            images = pipline_res.images
            movie = pipline_res.movie
            # print(f"movie: {type(movie)}, {type(movie[-1])}, images: {type(images)}")
            print(f"pipeline: {pipeline}")
            print(type(images), images.shape, images.max(), images.min())
            # <class 'numpy.ndarray'> (16, 32, 32, 3) 1.0 0.037254035
            torch.save(torch.from_numpy(images), generated_imgpt_savepath)
            # # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
            images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
            # init_images = [Image.fromarray(image) for image in np.squeeze((movie[999] * 255).round().astype("uint8"))]
            # # Make a grid out of the images
            image_grid = make_grid(images, rows=4, cols=4)
            # init_image_grid = make_grid(init_images, rows=4, cols=4)
            if save_gif:
                sam_obj = Samples(samples=np.array(movie), save_dir=f"{sample_folder}/sample.gif")
            # # Save the images
            image_grid.save(f"{sample_folder}/sample_grid{clip_opt}.png")
            # init_image_grid.save(f"{sample_folder}/sample_grid{clip_opt}_sample_Tminus1.png")

            # 以下保存movies
            # timesteps = pipeline.scheduler.timesteps
            # for i, t in enumerate(timesteps):
            #     movie_img = [Image.fromarray(image) for image in np.squeeze((movie[t] * 255).round().astype("uint8"))]
            #     movie_grid = make_grid(movie_img, rows=4, cols=4)
            #     movie_grid.save(f"{sample_folder}/movie_grid{clip_opt}_sample_{t}.png")


        with torch.no_grad():
            noise = torch.randn([8, ] + noise_shape).to(device)  # 16 for cifar10, 4 for celebahq256
            init = noise + trigger
            gen_samples(init=init)

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
    loss = torch.nn.functional.l1_loss(noise, output)  # l(x,y)=mean(x-y)
    return loss

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

    if config['dataset'] == "CIFAR10":
        config['eval_max_batch'] = 128
    else:
        config['eval_max_batch'] = 12

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
        pipeline = DDPMPipeline(unet=unet, scheduler=noise_sched)
        batch_sampling_save(sample_n=num_backdoor, num_inference_steps=1000, ddim_eta=None,
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
    # for R_coef_T in np.round(np.arange(0, 1.1, 0.1), 1):
        if str(R_coef_T) not in score_dict[model_name].keys():
            score_dict[model_name][str(R_coef_T)] = {}
        trigger_filename = f'{dir_path}/{file_name}/inverted_trigger/{model_name}_trigger_{R_coef_T}.pt'
        print(f"trigger filename: {trigger_filename}")
        if not os.path.isdir(os.path.dirname(trigger_filename)):
            os.makedirs(os.path.dirname(trigger_filename), exist_ok=True)

        if not os.path.isfile(trigger_filename): 
            pipeline = DDPMPipeline.from_pretrained(args.model_path).to(device)
            scheduler = pipeline.scheduler
            unet = pipeline.unet.eval()
            unet = unet.cuda()
            print(f"scheduler: {scheduler}")
            if scheduler.timesteps == None:
                scheduler.set_timesteps(num_inference_steps=1000)

            noise_shape = [unet.in_channels, unet.sample_size, unet.sample_size]
            if noise_shape[-1] == 256:
                bs = 8
            elif noise_shape[-1] == 64:
                bs = 36
            else:
                bs = 100  # cifar10
            noise = torch.randn([bs, ] + noise_shape).cuda()
            # noise = torch.randn([bs, ] + noise_shape)
            T = scheduler.num_train_timesteps-1
            print(T)

            print('R_coef_T', R_coef_T)

            trigger = -torch.rand([1, ] + noise_shape).cuda()  
            # trigger = -torch.rand([1, ] + noise_shape) 
            trigger.requires_grad_(True)
            optimizer = optim.Adam([trigger, ], lr=0.1)

            num_epochs = 100
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                unet.zero_grad()

                trigger_noise = noise + trigger
                model_output = unet(trigger_noise, T).sample
                loss = trigger_loss(trigger * R_coef_T, model_output)

                loss.backward()
                optimizer.step()
                print(f'{epoch} loss: {loss.item()}, R_coef_T: {R_coef_T if isinstance(R_coef_T, float) else R_coef_T.item()}')

            if not isinstance(R_coef_T, float):
                R_coef_T = R_coef_T.item()
            with torch.no_grad():
                # trigger_noise = noise + trigger
                # model_output = unet(trigger_noise, T).sample
                # save_image(trigger*0.5+0.5, './tmp_trigger.png')
                # save_image(model_output.mean(0).unsqueeze(0)*0.5+0.5, './tmp_output.png')
                # torch.save(trigger.cpu(), f'./{ckpt}_trigger.pt')
                torch.save(trigger.cpu(), trigger_filename)
        else:
            trigger = torch.load(trigger_filename, map_location='cuda:0')

        trigger = trigger.to(device)

        # sample with trigger, observe the effects of trigger
        sample_folder = os.path.join(dir_path, file_name, 'generated_img_with_trigger', f'{model_name}_inverted_{R_coef_T}')
        print(f"sample folder: {sample_folder}")
        sample_with_trigger(args, trigger, sample_folder, str(R_coef_T), score_json_path=score_json_path, save_res_dict=True)
        # eval_samples(args, str(R_coef_T), model_name=model_name, sample_folder=sample_folder, score_json_path=score_json_path)

        # measure with the synthesized trigger
        # measure_folder = os.path.join(dir_path, file_name, 'measure_with_generated_trigger', f'{model_name}_inverted_{R_coef_T}')
        # print(f"measure folder: {measure_folder}")
        # measure_trigger(args, trigger, measure_folder, str(R_coef_T), num_backdoor=1000, score_json_path=score_json_path, save_res_dict=True)


def compute_uniformity(images):
    if images.shape[-1] == 3:
        # last channle is rgb
        images = rearrange(images, 'b h w c -> b c h w')

    images1 = repeat(images, 'b c h w -> (b tile) c h w', tile=len(images))
    images2 = repeat(images, 'b c h w -> (tile b) c h w', tile=len(images))

    percept = LPIPS(replace_pooling=True, reduction="none")
    loss = percept(images1, images2).view(len(images), len(images))  # .mean()
    loss = torch.sort(loss, dim=1)[0]
    skip_cnt = 4
    loss = loss[:, skip_cnt:-skip_cnt]
    loss = loss.mean(dim=1)
    loss = torch.sort(loss)[0]
    loss = loss[skip_cnt:-skip_cnt].mean()

    return loss.item()


from torchmetrics.image import TotalVariation


def compute_tvloss(images):
    if images.shape[-1] == 3:
        # last channle is rgb
        images = rearrange(images, 'b h w c -> b c h w')

    tv = TotalVariation(reduction='mean').cuda()

    return tv(images).item()


if __name__ == '__main__':
    print("elijah baddiff")
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_tvloss', action='store_true',
                        help='compute tv loss instead of uniformity')  # TV loss和uniformity loss用于后门检测，而非trigger inversion
    parser.add_argument('--model_path', help='checkpoint')
    parser.add_argument('--dataset', type=str, help='datasets used by the model', default="cifar10")
    parser.add_argument('--batch', type=int, help='batchsize of computing loss and sampling', default=16)
    parser.add_argument('--eval', action='store_true', help='whether to evaluate the efficiency of synthesized trigger')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    score_dict = {}
    opt_r(args)

# python elijah_helper_baddiff.py --model_path D:/BackdoorDiff-DS/BadDiff/DDPM1-CIFAR10/res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-SHOE
# python elijah_helper_baddiff.py --model_path D:\BackdoorDiff-DS\Clean\DDPM3-CELEBAHQ256\res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep50_c1.0_p0.0_GLASSES-CAT

