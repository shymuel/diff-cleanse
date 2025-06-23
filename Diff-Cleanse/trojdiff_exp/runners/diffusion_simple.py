import os
import logging
import time
import glob

import numpy as np
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets_local import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
import torchvision.utils as tvu
from tqdm import tqdm

from PIL import Image
import copy
import torchvision.transforms as T


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
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
        miu = Image.open(args.miu_path)  # trigger
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
        coef_miu = torch.sqrt(1. - alphas_cumprod_prev) * betas - (1. - alphas_cumprod_prev) * torch.sqrt(
            alphas) * k_t
        self.coef_miu = coef_miu

        target_img = Image.open('images/mickey.png')
        target_img = transform(target_img)  # [0,1]
        self.target_img = target_img  # (3,32,32)

        self.build_model()

    def build_model(self):
        args, config = self.args, self.config
        model = Model(config)

        if args.restore_from is not None and os.path.isfile(args.restore_from):
            ckpt = args.restore_from
            print("Loading checkpoint {}".format(ckpt))
            states = torch.load(ckpt, map_location=self.config.device)
            if isinstance(states, torch.nn.Module):  # 因为训练没用ema，所以states不是列表，而是一个Model了
                model = torch.load(ckpt, map_location='cpu')
            elif isinstance(states, list):  # pruned model and training states
                model = states[0]
                if args.use_ema and self.config.model.ema:
                    print("Loading EMA")
                    ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                    ema_helper.register(model)
                    ema_helper.load_state_dict(states[-1])
                    ema_helper.ema(model)
                    self.ema_helper = ema_helper
                else:
                    self.ema_helper = None
            else:
                raise NotImplementedError
            self.model = model
        elif not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                ckpt = os.path.join(self.args.log_path, "ckpt.pth")
                states = torch.load(
                    ckpt,
                    map_location=self.config.device,
                )
            else:
                ckpt = os.path.join(
                    self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                )
                states = torch.load(
                    ckpt,
                    map_location=self.config.device,
                )
            print("Loading checkpoint {}".format(ckpt))

            if isinstance(states[0], torch.nn.Module):
                model = states[0]
                model = model.to(self.device)
            else:
                model = model.to(self.device)
                model.load_state_dict(states[0], strict=True)

            if args.use_ema and self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
                self.ema_helper = ema_helper
            else:
                self.ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == "CELEBA":
                name = 'celeba'
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            states = torch.load(ckpt, map_location=self.device)
            if isinstance(states, (list, tuple)):
                model.load_state_dict(states[0])
            else:
                model.load_state_dict(states)
            model.to(self.device)
        self.model = model

        model_f = Model(config)  # 这里两个模型的形状结构应该不同,model_f不需要更新参数，所以不需要emahelper
        if args.teacher_model_path is not None and os.path.isfile(args.teacher_model_path):  # 加载教师模型
            print("Loading teacher checkpoint {}".format(args.teacher_model_path))
            ckpt_f = args.teacher_model_path
            states = torch.load(ckpt_f, map_location='cpu')
            states[0] = {k.partition('module.')[2]: v for k, v in states[0].items()}

            if isinstance(states[0], torch.nn.Module):
                model_f = states[0]
                model_f = model_f.to(self.device)
            else:
                model_f = model_f.to(self.device)
                model_f.load_state_dict(states[0], strict=True)

            if args.use_ema and self.config.model.ema:
                print("Loading EMA")
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
                self.ema_helper = ema_helper
            else:
                self.ema_helper = None

            self.model_f = model_f
        else:
            self.model_f = None

        if self.model_f != None:
            pass


    def train(self):  # finetune
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        np.random.seed(0)
        from torch.utils.data import Subset
        indices = np.random.choice(range(len(dataset)), 5000, replace=False)
        dataset = Subset(dataset, indices)

        train_loader = data.DataLoader(dataset, batch_size=100, shuffle=True,
            num_workers=config.data.num_workers)
        model = self.model
        model = model.to(self.device)

        # model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if args.use_ema and self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        model.eval()
        if self.model_f != None:
            print("model f is not None")
            model_f = self.model_f
            model_f = model_f.to(self.device)
            model_f.eval()
        with torch.no_grad():
            self.sample_fid(model, n_samples=100)
            self.sample_fid_bd(model, n_samples=16)

        for epoch in range(start_epoch, 40):
            print(f"Epoch: {epoch}")
            for i, (x, y) in enumerate(tqdm(train_loader)):
                n = x.size(0)
                model.train()
                step += 1
                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas
                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b, model_f=model_f)

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                except Exception:
                    pass
                optimizer.step()

                if args.use_ema and self.config.model.ema:
                    ema_helper.update(model)

            model.zero_grad()

            # if epoch == 0 or (epoch+1) % 5 == 0:
            if (epoch + 1) % 40 == 0:
                states = [model]
                if args.use_ema and self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(states, os.path.join(self.args.log_path, f"unet_pruned_{epoch}.pth"))

                # Sampling for visualization
                model.eval()
                with torch.no_grad():
                    n = 100
                    x = torch.randn(n, config.data.channels, config.data.image_size,
                        config.data.image_size, device=self.device)
                    x = self.sample_image(x, model)
                    x = inverse_data_transform(config, x)
                    grid = tvu.make_grid(x)
                    tvu.save_image(grid, os.path.join(args.log_path, 'vis', f'epoch_{epoch}.png'))

    def sample(self):
        model = self.model  # Model(self.config)
        model.to(self.device)
        # model = torch.nn.DataParallel(model)
        model.eval()

        if self.args.fid:
            self.sample_fid(model, n_samples=10000)
            self.sample_fid_bd(model, n_samples=16)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model, n_samples=64):
        import torch
        torch.manual_seed(0)
        import random
        random.seed(0)
        import numpy as np
        np.random.seed(0)

        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"image folder: {self.args.image_folder}")
        print(f"starting from image {img_id}")
        total_n_samples = n_samples
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        print("n: ", config.sampling.batch_size)
        with torch.no_grad():
            for _ in tqdm(
                    range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{img_id}.png"))
                    img_id += 1

    def sample_fid_bd(self, model, n_samples=16):
        import torch
        torch.manual_seed(0)
        import random
        random.seed(0)
        import numpy as np
        np.random.seed(0)

        config = self.config
        if self.args.use_pretrained:
            image_folder = self.args.image_folder + '_pretrained_bd'
        else:
            image_folder = self.args.image_folder + '_bd'
        if self.args.eta != 1:  # ddim
            image_folder = image_folder + '_ddim_eta_' + str(self.args.eta)

        image_folder = self.args.image_folder + '_bd'
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        print(f"image folder bd: {image_folder}")
        img_id = len(glob.glob(f"{image_folder}/*"))
        # img_id = 0
        print(f"starting from image {img_id}")
        total_n_samples = n_samples
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        n_rounds = 1

        with torch.no_grad():
            for _ in tqdm(range(n_rounds), desc="Generating image samples for SSIM and MSE evaluation."):
                n = 16
                x = torch.randn(n, config.data.channels, config.data.image_size, config.data.image_size,
                                device=self.device)
                miu = torch.stack([self.miu.to(self.device)] * n)  # (batch,3,32,32)
                tmp_x = x.clone()  # 原图
                x = self.args.gamma * x + miu  # N(miu,I)
                if self.args.trigger_type == 'patch':
                    tmp_x[:, :, -self.args.patch_size:, -self.args.patch_size:] = x[:, :, -self.args.patch_size:,
                                                                                  -self.args.patch_size:]  # 只有patch对应的部分修改了
                    x = tmp_x  # 赋值给x

                x = self.sample_image_bd(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(x[i], os.path.join(image_folder, f"{img_id}.png"))
                    img_id += 1

    def sample_sequence(self, model):
        '''
        保存生成的图片的每个通道
        '''
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

    def sample_interpolation(self, model):
        '''使用球面线性插值（Slerp）在两个随机向量之间进行图像插值。'''
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
                xs.append(self.sample_image(x[i: i + 8], model))
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
                seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps)** 2)
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
                seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps)** 2)
                seq = [int(s) for s in list(seq)]
                print(seq)
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
                seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps)** 2)
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
                seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps) ** 2)
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

    def test(self, clean_image_folder, backdoor_image_folder):
        from fid_score import fid
        from evaluation_tools import ssim, mse

        dataset_img_dir = "E:\\12_diffusionmodel\\BadDiffusion\\measure\\CIFAR10"
        fid_sc = float(fid(path=[dataset_img_dir, clean_image_folder], device="cuda", num_workers=1))
        mse_sc = mse(backdoor_path=backdoor_image_folder, device="cuda", num_workers=1)
        ssim_sc = ssim(backdoor_path=backdoor_image_folder, device="cuda", num_workers=1)
        print(f"fic score: {fid_sc}; mse score: {mse_sc}, ssim score: {ssim_sc}")
        return fid_sc, mse_sc, ssim_sc
