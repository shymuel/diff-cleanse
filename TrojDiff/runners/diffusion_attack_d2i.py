import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from torchmetrics import StructuralSimilarityIndexMeasure
from torch import nn

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses_attack_d2dout import loss_registry
from functions.losses import loss_registry as benign_loss_registry
from datasets import get_dataset, get_targetset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu
import torchvision.transforms as T
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize


from PIL import Image

from typing import List, Union
from fid_score import fid
import pathlib
from joblib import Parallel, delayed
import json
import re


def cycle(dl):
    while True:
        for data in dl:
            yield data


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def get_batch_sizes(sample_n: int, max_batch_n: int):
    if sample_n > max_batch_n:
        replica = sample_n // max_batch_n
        residual = sample_n % max_batch_n
        batch_sizes = [max_batch_n] * (replica) + ([residual] if residual > 0 else [])
    else:
        batch_sizes = [sample_n]
    return batch_sizes

def batchify(xs, max_batch_n: int):
    # sample_n: int = len(xs)
    # if sample_n > max_batch_n:
    #     replica = sample_n // max_batch_n
    #     residual = sample_n % max_batch_n
    #     batch_sizes = [max_batch_n] * (replica) + ([residual] if residual > 0 else [])
    # else:
    #     batch_sizes = [sample_n]
    batch_sizes = get_batch_sizes(sample_n=len(xs), max_batch_n=max_batch_n)

    # print(f"xs len(): {len(xs)}")
    # print(f"batch_sizes: {batch_sizes}, max_batch_n: {max_batch_n}")
    res: List = []
    cnt: int = 0
    for i, bs in enumerate(batch_sizes):
        res.append(xs[cnt:cnt + bs])
        cnt += bs
    return res

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
        batch_operator: callable = Metric.get_batch_operator(a=a, b=b)

        def metric(x, y):
            mse: torch.Tensor = nn.MSELoss(reduction='none')(x, y).mean(dim=[i for i in range(1, len(x.shape))])
            # print(f"MSE: {mse.shape}")
            return mse

        return float(batch_operator(a=a, b=b, max_batch_n=max_batch_n, fn=metric))

    @staticmethod
    def mse_thres_batch(a: torch.Tensor, b: torch.Tensor, thres: float, max_batch_n: int):
        batch_operator: callable = Metric.get_batch_operator(a=a, b=b)

        def metric(x, y):
            # print(f"x: {x.shape}, y: {y.shape}")
            # print(f"Mean Dims: {[i for i in range(1, len(x))]}")
            probs: torch.Tensor = nn.MSELoss(reduction='none')(x, y).mean(dim=[i for i in range(1, len(x.shape))])
            mse_thres: torch.Tensor = torch.where(probs < thres, 1.0, 0.0)
            # print(f"MSE Threshold: {mse_thres.shape}")
            return mse_thres

        return float(batch_operator(a=a, b=b, max_batch_n=max_batch_n, fn=metric))

    @staticmethod
    def ssim_batch(a: torch.Tensor, b: torch.Tensor, device: str, max_batch_n: int):
        batch_operator: callable = Metric.get_batch_operator(a=a, b=b)

        def metric(x, y):
            ssim: torch.Tensor = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)(x, y)
            if len(ssim.shape) < 1:
                ssim = ssim.unsqueeze(dim=0)
            # print(f"SSIM: {ssim.shape}")
            return ssim

        return float(batch_operator(a=a, b=b, max_batch_n=max_batch_n, fn=metric))


class ImagePathDataset(torch.utils.data.Dataset):
    IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}

    # TRANSFORM = [transforms.ToTensor()]

    def __init__(self, path, transforms=None, njobs: int = -1):
        self.__path = pathlib.Path(path)
        self.__files = sorted([file for ext in ImagePathDataset.IMAGE_EXTENSIONS
                               for file in self.__path.glob('*.{}'.format(ext))])
        self.__transforms = transforms
        self.__njobs = njobs

    def __len__(self):
        return len(self.__files)

    def read_imgs(self, paths: Union[str, List[str]]):
        # to_tensor = lambda path: transforms.ToTensor()(Image.open(path).copy().convert('RGB'))
        # trans_ls = [transforms.Lambda(to_tensor)]
        trans_ls = [transforms.Lambda(ImagePathDataset.__read_img)]
        if self.__transforms != None:
            trans_ls += self.__transforms

        if isinstance(paths, list):
            if self.__njobs == None:
                print(f"n-jobs: {self.__njobs}, Read images sequentially")
                imgs = [Compose(trans_ls)(path) for path in paths]
            else:
                print(f"n-jobs: {self.__njobs}, Read images concurrently")
                imgs = list(Parallel(n_jobs=self.__njobs)(delayed(Compose(trans_ls))(path) for path in paths))
            print(f"n-jobs: {self.__njobs}, Reading Images done")
            return torch.stack(imgs)
        return transforms.ToTensor()(Image.open(paths).convert('RGB'))

    def fetch_slice(self, start: int, end: int, step: int = 1):
        read_ls: List[str] = list(set(self.__files))[slice(start, end, step)]
        # return Compose([transforms.Lambda(self.read_imgs)])(read_ls)
        return self.read_imgs(read_ls)

    @staticmethod
    # @lru_cache(1000)
    def __read_img(path):
        return transforms.ToTensor()(Image.open(path).copy().convert('RGB'))

    def __getitem__(self, slc):
        # img = Compose([transforms.Lambda(self.read_imgs)])(self.__files[slc])
        # return img
        return self.read_imgs(list(set(self.__files))[slc])



class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # attack
        miu = Image.open(args.miu_path)
        transform = T.Compose([
            T.Resize((config.data.image_size, config.data.image_size)),
            T.ToTensor()
        ])
        miu = transform(miu)  # [0,1]
        miu = data_transform(self.config, miu)  # [-1,1]
        miu = miu * (1 - args.gamma)  # [-0.5,0.5]
        self.miu = miu  # (3,32,32)

        k_t = torch.randn_like(betas)
        for ii in range(config.diffusion.num_diffusion_timesteps):
            tmp_sum = torch.sqrt(1. - alphas_cumprod[ii])
            tmp_alphas = torch.flip(alphas[:ii + 1], [0])
            for jj in range(1, ii + 1):
                tmp_sum -= k_t[ii - jj] * torch.sqrt(torch.prod(tmp_alphas[:jj]))
            k_t[ii] = tmp_sum
        coef_miu = torch.sqrt(1. - alphas_cumprod_prev) * betas - (1. - alphas_cumprod_prev) * torch.sqrt(alphas) * k_t
        self.coef_miu = coef_miu

        target_img = Image.open('D:/Diff-Cleanse-draft/trojdiff/images/mickey.png')
        target_img = transform(target_img)  # [0,1]
        self.target_img = target_img  # (3,32,32)


    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        # base dataset
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        model = Model(config)

        model = model.to(self.device)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            # attack: resume from pre-trained model
            if self.config.data.dataset == "CIFAR10" or self.config.data.dataset == "LSUN":
                if self.config.data.dataset == "CIFAR10":
                    name = "cifar10"
                elif self.config.data.dataset == "LSUN":
                    name = f"lsun_{self.config.data.category}"
                else:
                    raise ValueError
                ckpt = get_ckpt_path(f"ema_{name}")
                print("Loading checkpoint {}".format(ckpt))
                states = torch.load(ckpt, map_location=self.device)
                model.load_state_dict(states)
                model = torch.nn.DataParallel(model)
                if self.config.model.ema:
                    ema_helper.load_state_dict(states)
            elif self.config.data.dataset == "CELEBA":
                ckpt = 'D:/01_dm/ddim/ddim/ckpt_4020000.pth'
                states = torch.load(ckpt, map_location=self.device)[4]
                model.load_state_dict(states)
                model = torch.nn.DataParallel(model)
                if self.config.model.ema:
                    ema_helper.load_state_dict(states)

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                # target data
                target_bs = int(x.shape[0]*0.1)
                x_tar = torch.stack([self.target_img] * target_bs)  # (batch,3,32,32)
                y_tar = torch.ones(target_bs) * 1000  # (batch)
                x = torch.cat([x, x_tar], dim=0)
                y= torch.cat([y, y_tar], dim=0)

                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1
                y = y.to(self.device)

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                loss = loss_registry[config.model.type](model, x, y, t, e, b, self.miu, self.args)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)


                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()


    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            print("Loading model...")
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10" or self.config.data.dataset == "LSUN":
                if self.config.data.dataset == "CIFAR10":
                    name = "cifar10"
                elif self.config.data.dataset == "LSUN":
                    name = f"lsun_{self.config.data.category}"
                else:
                    raise ValueError
                ckpt = get_ckpt_path(f"ema_{name}")
                print("Loading checkpoint {}".format(ckpt))
                model.load_state_dict(torch.load(ckpt, map_location=self.device))
                model.to(self.device)
                model = torch.nn.DataParallel(model)

            elif self.config.data.dataset == "CELEBA":
                ckpt = '/home/username/ddim/saved/celeba_ckpt.pth'
                states = torch.load(ckpt, map_location=self.device)[4]
                model.load_state_dict(states)
                model.to(self.device)
                model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid_bd(model, total_n_samples=10000)
            self.sample_fid(model, total_n_samples=10000)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence_bd(model)
            # self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model, image_save_path=None, total_n_samples=16):
        config = self.config
        if image_save_path is None:
            if self.args.use_pretrained:
                image_folder = self.args.image_folder + '_pretrained'
            else:
                image_folder = self.args.image_folder + '_ckpt' + str(self.config.sampling.ckpt_id)
            if self.args.eta != 1: # ddim
                image_folder = image_folder + '_ddim_eta_' + str(self.args.eta)
            image_folder = image_folder + '_' + self.args.dataset
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
        else:
            image_folder = image_save_path

        os.makedirs(image_folder, exist_ok=True)

        img_id = len(glob.glob(f"{image_folder}/*"))
        # img_id = 0
        print(f"starting from image {img_id}")
        total_n_samples = total_n_samples
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size + 1

        with torch.no_grad():
            for n_round in tqdm.tqdm(range(n_rounds), desc="Generating image samples for FID evaluation."):
                if n_round == n_rounds - 1:
                    n = (total_n_samples - img_id) % config.sampling.batch_size
                    if n == 0:
                        break
                else:
                    n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_fid_bd(self, model, image_save_path=None, total_n_samples=16):
        config = self.config
        if image_save_path is None:
            if self.args.use_pretrained:
                image_folder = self.args.image_folder + '_pretrained_bd'
            else:
                image_folder = self.args.image_folder + '_ckpt' + str(self.config.sampling.ckpt_id) + '_bd'
            if self.args.eta != 1: # ddim
                image_folder = image_folder + '_ddim_eta_' + str(self.args.eta)
            image_folder = image_folder + '_' + self.args.dataset
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
        else:
            image_folder = image_save_path

        os.makedirs(image_folder, exist_ok=True)

        img_id = len(glob.glob(f"{image_folder}/*"))
        # img_id = 0
        print(f"starting from image {img_id}")
        total_n_samples = total_n_samples
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size + 1

        with torch.no_grad():
            for n_round in tqdm.tqdm(range(n_rounds), desc="Generating image samples for backdoor metrics."):
                if n_round == n_rounds - 1:
                    n = (total_n_samples - img_id) % config.sampling.batch_size
                    if n == 0:
                        break
                else:
                    n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                miu = torch.stack([self.miu.to(self.device)] * n)  # (batch,3,32,32)
                tmp_x = x.clone()
                x = self.args.gamma * x + miu  # N(miu,I)
                if self.args.trigger_type == 'patch':
                    tmp_x[:, :, -self.args.patch_size:, -self.args.patch_size:] = x[:, :, -self.args.patch_size:, -self.args.patch_size:]
                    x = tmp_x

                x = self.sample_image_bd(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_sequence_bd(self, model):
        config = self.config

        n = 4
        x = torch.randn(
            n,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        miu = torch.stack([self.miu.to(self.device)] * n)  # (batch,3,32,32)
        tmp_x = x.clone()
        x = self.args.gamma * x + miu  # N(miu,I)
        if self.args.trigger_type == 'patch':
            tmp_x[:, :, -self.args.patch_size:, -self.args.patch_size:] = x[:, :, -self.args.patch_size:,
                                                                          -self.args.patch_size:]
            x = tmp_x

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            x, _ = self.sample_image_bd(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]
        x[0] = x[0].cpu()
        x = torch.stack(x) # [100, 4, 3, 32, 32]

        for i in range(x.shape[1]):
            tvu.save_image(x[1:,i], os.path.join('/home/username/ddim/images/d2i', f"img{i}_process.png"), nrow=50)#[100,3,32,32]
            # for j in range(x.shape[0]):
            #     tvu.save_image(
            #         x[j,i], os.path.join('/home/username/ddim/images/d2i', f"img{i}_timestep{j}.png")#[3,32,32]
            #     )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def sample_image_bd(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps_bd

            xs = generalized_steps_bd(x, seq, model, self.betas, self.miu, self.coef_miu, self.args, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps_bd

            x = ddpm_steps_bd(x, seq, model, self.betas, self.miu, self.coef_miu, self.args)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def measure(self, num_clean, num_backdoor):
        # load the model
        model = Model(self.config)

        if self.args.ckpt_path:
            model_dir = os.path.join(self.args.ckpt_path, "model")
            pattern = re.compile(r'^ckpt_(\d+)\.pth$')
            max_num = -1
            latest_file = None
            for file_name in os.listdir(model_dir):
                match = pattern.match(file_name)
                if match:
                    num = int(match.group(1))
                    if num > max_num: 
                        max_num = num
                        latest_file = file_name
            print(f"Loading model: {latest_file}")
            states = torch.load(os.path.join(model_dir, latest_file), map_location=self.config.device,)
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        model.eval()

        # create files to store the sampled images
        if self.args.ckpt_path:  # file path for BackdoorDiff
            clean_image_folder = os.path.join(self.args.ckpt_path, "measure", f"clean_{latest_file[:-4]}_{num_clean}")
            bd_image_folder = os.path.join(self.args.ckpt_path, "measure", f"backdoor_{latest_file[:-4]}_{num_backdoor}")
        elif self.args.use_pretrained:  # file path for default mode
            clean_image_folder = self.args.image_folder + '_pretrained'
            bd_image_folder = self.args.image_folder + '_pretrained_bd'
        else:
            clean_image_folder = self.args.image_folder + '_ckpt' + str(self.config.sampling.ckpt_id)
            bd_image_folder = self.args.image_folder + '_ckpt' + str(self.config.sampling.ckpt_id) + '_bd'

        if self.args.eta != 1:  # sample with ddim
            clean_image_folder = clean_image_folder + '_ddim_eta_' + str(self.args.eta)
            bd_image_folder = bd_image_folder + '_ddim_eta_' + str(self.args.eta)

        clean_image_folder = clean_image_folder + "_" + self.args.dataset
        bd_image_folder = bd_image_folder + "_" + self.args.dataset
        print(clean_image_folder, bd_image_folder)

        os.makedirs(clean_image_folder, exist_ok=True)
        os.makedirs(bd_image_folder, exist_ok=True)

        num_clean_images = len(os.listdir(clean_image_folder))
        num_bd_images = len(os.listdir(bd_image_folder))
        print(f"num clean images: {num_clean_images}, num bd images: {num_bd_images}")
        if num_clean_images < num_clean:
            self.sample_fid(model, clean_image_folder)
        if num_bd_images < num_backdoor:
            self.sample_fid_bd(model, bd_image_folder)

        sched_type = 'DDIM' if self.args.eta != 1 else 'DDPM'
        fid_key = f"FID_{sched_type}_{num_clean}"
        mse_key = f"MSE_{sched_type}_{num_backdoor}"
        ssim_key = f"SSIM_{sched_type}_{num_backdoor}"
        asr_key = f"ASR_{sched_type}_{num_backdoor}"

        if self.args.dataset == "cifar10":
            dataset_img_dir = "D:\\00_dataset\\cifar10.npz"
        elif self.args.dataset == "celeba":
            dataset_img_dir = "D:\\00_dataset\\celeba.npz"

        # calculate fid
        print(f"fid: {dataset_img_dir} / {clean_image_folder}")
        fid_sc = float(fid(path=[dataset_img_dir, clean_image_folder], device=self.device, num_workers=1,
                           batch_size=self.config.sampling.batch_size))
        print(f"FID Score: {fid_sc}")

        # calculate backdoor metrics
        gen_backdoor_target = ImagePathDataset(path=bd_image_folder, njobs=1)[:].to(self.device)
        reps = ([len(gen_backdoor_target)] + ([1] * (len(self.target_img.shape))))
        backdoor_target = torch.squeeze((self.target_img.repeat(*reps))).to(self.device)

        mse_sc = Metric.mse_batch(a=gen_backdoor_target, b=backdoor_target, max_batch_n=self.config.sampling.batch_size)
        ssim_sc = Metric.ssim_batch(a=gen_backdoor_target, b=backdoor_target, max_batch_n=self.config.sampling.batch_size,
                                    device=self.device)
        asr_sc = Metric.mse_thres_batch(a=gen_backdoor_target, b=backdoor_target,
                                        thres=0.01, max_batch_n=self.config.sampling.batch_size)

        print(f"MSE: {mse_sc}, SSIM: {ssim_sc}, ASR: {asr_sc}")

        score_dict = {fid_key: fid_sc, mse_key: mse_sc, ssim_key: ssim_sc,
            asr_key: asr_sc  
        }

        output_json_path = os.path.join(self.args.ckpt_path, 'score.json')
        with open(output_json_path, 'w') as json_file:
            json.dump(score_dict, json_file, indent=4)

        print(f"Scores saved to {output_json_path}")

    def get_model(self, before_fix=True, before_merge=True):
        # for elijah trigger inversion
        model = Model(self.config)

        model_ckpt_dir = self.args.ckpt_path
        if self.args.remove_backdoor and before_fix:
            model_ckpt_dir = model_ckpt_dir.replace('_remove_backdoor', '')

        if self.args.merge_backdoor and before_merge:
            model_ckpt_dir = model_ckpt_dir.replace('_merge_backdoor', '')

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                # model_pt_file = os.path.join(model_ckpt_dir, "ckpt.pth")
                model_pt_file = model_ckpt_dir
                states = torch.load(
                    model_pt_file,
                    map_location=self.config.device,
                )
            else:
                # model_pt_file = os.path.join(model_ckpt_dir, f"ckpt_{self.config.sampling.ckpt_id}.pth")
                model_pt_file = model_ckpt_dir
                states = torch.load(
                    model_pt_file,
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            print(f'Loading model from {model_pt_file}')
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10" or self.config.data.dataset == "LSUN":
                if self.config.data.dataset == "CIFAR10":
                    name = "cifar10"
                elif self.config.data.dataset == "LSUN":
                    name = f"lsun_{self.config.data.category}"
                else:
                    raise ValueError
                ckpt = get_ckpt_path(f"ema_{name}")
                print("Loading checkpoint {}".format(ckpt))
                model.load_state_dict(torch.load(ckpt, map_location=self.device))
                model.to(self.device)
                model = torch.nn.DataParallel(model)

            elif self.config.data.dataset == "CELEBA":
                ckpt = '/home/username/ddim/saved/celeba_ckpt.pth'
                states = torch.load(ckpt, map_location=self.device)[4]
                model.load_state_dict(states)
                model.to(self.device)
                model = torch.nn.DataParallel(model)

        model.eval()
        return model

    def merge_backdoor(self):
        self.remove_backdoor(trigger=None)

    def remove_backdoor(self, trigger=None):
        args, config = self.args, self.config
        # tb_logger = self.config.tb_logger

        if trigger is None:
            assert args.merge_backdoor, 'merge the backdoor sampling and benign sampling'
            trigger = 0

        # base dataset
        # NOTE: to reduece to 10% dataset
        dataset, test_dataset = get_dataset(args, config)
        logging.info(f'dataset length: {len(dataset)}')
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        model = Model(config)

        model = model.to(self.device)

        optimizer = get_optimizer(self.config, model.parameters())

        start_epoch, step = 0, 0

        model_ckpt_dir = self.args.ckpt_path
        # if self.args.remove_backdoor:
        #     model_ckpt_dir = model_ckpt_dir.replace('_remove_backdoor', '')

        # if isinstance(trigger, int):
            # when trigger is 0, we merge the backdoor and benign from original backdoored model
            # otherwise, it means we are trying to remove backdoor from merge backdoor
            #            where we need to set both --remove_backdoor and --merge_backdoor
            # model_ckpt_dir = model_ckpt_dir.replace('_merge_backdoor', '')

        # attack: resume from pre-trained model
        # if getattr(self.config.sampling, "ckpt_id", None) is None:
        #     model_pt_file = os.path.join(model_ckpt_dir, "ckpt.pth")
        #     states = torch.load(
        #         model_pt_file,
        #         map_location=self.config.device,
        #     )
        # else:
        #     model_pt_file = os.path.join(model_ckpt_dir, f"ckpt_{self.config.sampling.ckpt_id}.pth")
        #     states = torch.load(
        #         model_pt_file,
        #         map_location=self.config.device,
        #     )

        model_pt_file = model_ckpt_dir
        states = torch.load(
            model_pt_file,
            map_location=self.config.device,
        )
        model = model.to(self.device)
        logging.info(f'Loading model from {model_pt_file}')
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            logging.info('using ema')
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None

        before_merge = not args.remove_backdoor

        @torch.no_grad()
        def get_frozen_model(before_fix=True, before_merge=before_merge):
            model = self.get_model(before_fix=before_fix, before_merge=before_merge)
            for p in model.parameters():
                p.requires_grad_(False)
            return model

        frozen_model = get_frozen_model()

        seq = range(0, self.num_timesteps)
        from functions.denoising import my_ddpm_one_step, my_benign_ddpm_one_step
        # define deshift loss

        def deshift_loss(noise, gen_mean_func_backdoor, gen_mean_func_benign, gen_mean_func_benign_frozen):
            # goal is to make sure the output of gen_mean_func_backdoor and gen_mean_func_benign is similar
            benign_prediction = gen_mean_func_benign(noise)
            with torch.no_grad():
                frozen_benign_prediction = gen_mean_func_benign_frozen(noise)
            if isinstance(trigger, int):
                assert trigger == 0
                backdoor_prediction = gen_mean_func_backdoor(noise)
            else:
                backdoor_prediction = gen_mean_func_backdoor(noise + trigger.to(noise.device))
            loss1 = torch.nn.functional.mse_loss(backdoor_prediction, frozen_benign_prediction)
            loss2 = torch.nn.functional.mse_loss(benign_prediction, frozen_benign_prediction)
            return loss1, loss2

        from tqdm import tqdm

        for epoch in range(start_epoch, self.config.training.n_epochs):
            print(f"epoch {epoch}")
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(tqdm(train_loader, desc="Training Progress", total=len(train_loader))):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1
                # y = y.to(self.device)

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                # loss0 = 0.

                # loss0 = loss_registry[config.model.type](model, x, y, t, e, b, self.miu, self.args)

                loss0 = benign_loss_registry[config.model.type](model, x, t, e, b)

                noise = e
                if not self.config.model.ema or isinstance(trigger, int):
                    # if not ema, use this loss
                    # if merge backdoor, use this loss
                    gen_mean_func_backdoor = my_ddpm_one_step(noise, seq, model, self.betas, self.miu, self.coef_miu,
                                                              self.args)
                    gen_mean_func_benign = my_benign_ddpm_one_step(noise, seq, model, self.betas, self.miu,
                                                                   self.coef_miu, self.args)
                    gen_mean_func_benign_frozen = my_benign_ddpm_one_step(noise, seq, frozen_model, self.betas,
                                                                          self.miu, self.coef_miu, self.args)
                else:
                    # remove backdoor
                    def gen_mean_func_benign(xT):
                        return model(xT, (torch.ones(n) * (self.num_timesteps - 1)).to(noise.device))

                    gen_mean_func_backdoor = gen_mean_func_benign

                    def gen_mean_func_benign_frozen(xT):
                        return frozen_model(xT, (torch.ones(n) * (self.num_timesteps - 1)).to(noise.device))

                loss1, loss2 = deshift_loss(e, gen_mean_func_backdoor, gen_mean_func_benign,
                                            gen_mean_func_benign_frozen)

                loss = loss0 + loss1 + loss2

                # tb_logger.add_scalar("loss", loss, global_step=step)
                # tb_logger.add_scalar("loss0", loss0, global_step=step)
                # tb_logger.add_scalar("loss1", loss1, global_step=step)
                # tb_logger.add_scalar("loss2", loss2, global_step=step)

                logging.info(
                    f"epoch: {epoch}, step: {step}, loss: {loss.item()}, loss0: {loss0 and loss0.item()},"
                    f"loss1: {loss1 and loss1.item()}, loss2: {loss2 and loss2.item()}, data time: {data_time / (i + 1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % 2000 == 0 or step == self.args.max_steps:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join("./Elijah_rm", "ckpt_{}.pth".format(step)),
                    )

                if step % self.config.training.validation_freq == 0 or step == 1 or step == self.args.max_steps:
                    self.sample_fid_bd(model, image_save_path=f"./elijah_rm/images_bd/{step}", total_n_samples=16)
                    self.sample_fid(model, image_save_path=f"./elijah_rm/images/{step}", total_n_samples=16)

                if step == self.args.max_steps:
                    return

                data_start = time.time()