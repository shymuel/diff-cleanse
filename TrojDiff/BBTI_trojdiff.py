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
from torch import nn

from runners.diffusion_attack import Diffusion

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses_attack_d2dout import loss_registry
from datasets import inverse_data_transform
import copy
from PIL import Image
from torchmetrics import StructuralSimilarityIndexMeasure

import math
import torchvision.transforms as T
from dataset import ImagePathDataset
import torch.nn.functional as F
import os.path as osp

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data.")
    parser.add_argument("--doc", type=str, default='ddpm', help="A string for documentation purpose. Will be the name of the log folder.")
    parser.add_argument("--comment", type=str, default="", help="A string for experiment comment")
    parser.add_argument("--verbose", type=str, default="info", help="Verbose level: info | debug | warning | critical")
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument("--sample", action="store_true", help="Whether to produce samples from the model")
    parser.add_argument("--reverse", action="store_true", help="Whether to reverse eigeering")
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
    parser.add_argument('--batch', type=int, default=16)

    # attack
    parser.add_argument('--attack', type=str, default='din', help="This arg is used to load models, not used for specified defense")
    parser.add_argument('--cond_prob', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--target_label', type=int, default=7)
    parser.add_argument('--miu_path', type=str, default='images/hello_kitty.png')
    parser.add_argument('--total_n_samples', type=int, default=50000)
    parser.add_argument('--trigger_type', type=str, default='blend')
    parser.add_argument('--patch_size', type=int, default=3)

    args = parser.parse_args()

    if args.attack == "din":
        model_type_path = "ddpm_attack"
    elif args.attack == "dout":
        model_type_path = "ddpm_attack_dout"
    else:
        model_type_path = "ddpm_attack_d2i"

    args.exp = os.path.join(model_type_path,
                            'ft_cond_prob_' + str(args.cond_prob) + '_gamma_' + str(
                                args.gamma) + '_target_label_' + str(args.target_label) + '_trigger_type_' + str(
                                args.trigger_type))  # attack
    if args.trigger_type == 'patch':
        args.exp = args.exp + '_size_' + str(args.patch_size)

    if not os.path.exists(args.exp):
        os.makedirs(args.exp)

    args.log_path = os.path.join(args.exp, "logs", args.doc)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)

    if args.resume_training:
        # if not args.resume_training:
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

        if args.sample:
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
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


def sample_image_bd(x, model, betas, miu, coef_miu, args, last=True):
    try:
        skip = args.skip
    except Exception:
        skip = 1

    num_timesteps = 1000

    if args.sample_type == "generalized":
        if args.skip_type == "uniform":
            skip = num_timesteps // args.timesteps
            seq = range(0, num_timesteps, skip)
        elif args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(num_timesteps * 0.8), args.timesteps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        from functions.denoising import generalized_steps

        xs = generalized_steps(x, seq, model, betas, eta=args.eta)
        x = xs
    elif args.sample_type == "ddpm_noisy":
        if args.skip_type == "uniform":
            skip = num_timesteps // args.timesteps
            seq = range(0, num_timesteps, skip)
        elif args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(num_timesteps * 0.8), args.timesteps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        from functions.denoising import ddpm_steps_bd

        x = ddpm_steps_bd(x=x, seq=seq, model=model, b=betas, miu=miu, coef_miu=coef_miu, args=args)
    else:
        raise NotImplementedError
    if last:
        x = x[0][-1]
    return x

def sample_bd(model, miu, config, args, betas, coef_miu, n_sample=16, device='cuda:0'):
    x = torch.randn(n_sample, 3, 32, 32, device=device)
    miu_t = torch.stack([miu.to(device)] * n_sample)
    print(f"miu shape: {miu_t.shape}, x shape: {x.shape}")
    tmp_x = x.clone()
    x = args.gamma * x + miu_t

    x = sample_image_bd(x, model, betas=betas, miu=miu, coef_miu=coef_miu, args=args)
    x = inverse_data_transform(config, x)
    print(x.shape)
    return x

def ste_round(x):  # round且保留梯度
    return torch.round(x) - x.detach() + x

def make_grid(images, rows, cols):
    w, h = images[0].size
    # print("image size", images[0].size)
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def calculate_entropy(logits):
    # Apply softmax to convert logits into probabilities
    probabilities = F.softmax(logits, dim=-1)
    # Calculate the entropy
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9),
                         dim=-1)  # Adding a small value to avoid log(0)
    return entropy.mean()  # Return the mean entropy if you want a scalar value

def calc_loss_entropy(target_set, residual_set, images):
    loss_x0_entropy_weight = 10
    loss_x0_entropy = 0

    entropy_threshold_max = 8.5
    entropy_threshold_min = 0.1

    for i in residual_set:
        entropy_loss_tmp = calculate_entropy(ste_round(images[i] * 255))  # 截断，去除高频噪声
        # print(f"i: {entropy_loss_tmp}")
        loss_x0_entropy += torch.relu(entropy_loss_tmp - entropy_threshold_max) + torch.relu(
            entropy_threshold_min - entropy_loss_tmp)

    if loss_x0_entropy.item() > 10:
        loss_x0_entropy_weight = 2

    return loss_x0_entropy_weight, loss_x0_entropy


def cosine_similarity(A, B):
    dot_product = torch.dot(A, B)
    norm_a = torch.norm(A)
    norm_b = torch.norm(B)
    return dot_product / (norm_a * norm_b)


import itertools
import networkx as nx

def calc_loss_similarity(args, image_features):
    combinations = list(itertools.combinations(range(args.batch), 2))
    target_set = []

    G = nx.Graph()
    for (a, b) in combinations:
        similarity = cosine_similarity(image_features[a].reshape(1, -1), image_features[b].reshape(1, -1))
        if similarity > 0.9:  # 0.7 for din and dout
            G.add_edge(a, b)

    cliques = list(nx.find_cliques(G))
    max_clique = max(cliques, key=len)

    flag = 0
    if max_clique > 0.5*len(image_features):
        flag = 1

    residual_set = []
    for i in range(args.batch):
        if not i in max_clique:
            residual_set.append(i)

    similarities = []
    similarities = []
    combinations = list(itertools.combinations(residual_set, 2))
    for (a, b) in combinations:
        similarity = cosine_similarity(image_features[a].reshape(1, -1), image_features[b].reshape(1, -1))
        similarities.append(similarity)
    loss_s = len(combinations) - torch.sum(torch.stack(similarities))  # f(x0)
    return flag, target_set, residual_set, loss_s


import clip


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

class InterpolateTransform:
    def __init__(self, size, mode='bilinear', align_corners=False):
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, img):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)  # 增加批次维度
        return F.interpolate(img, size=self.size, mode=self.mode, align_corners=self.align_corners)

transforms = Compose([
    InterpolateTransform(size=(224, 224), mode='bilinear', align_corners=False),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

def calc_mse(tensor1, tensor2):
    mse_distance = torch.mean((tensor1 - tensor2) ** 2)
    return mse_distance

def trigger_inversion(args, config, runner):
    from PIL import Image
    import itertools

    def get_raw_tensor(x):
        return nn.Tanh()(x)

    def element_wize_grad(x, y):
        grads = torch.zeros_like(x)
        if x.grad is not None:
            x.grad.zero_()

        shape_x = x.shape
        for i in range(shape_x[0]):
            for j in range(shape_x[1]):
                for k in range(shape_x[2]):
                    y[i, j, k].backward(retain_graph=True)
                    grads[i, j, k] = x.grad[i, j, k]
        if x.grad is not None:
            x.grad.zero_()
        return grads.to(device)

    current_file_path = __file__
    file_name = os.path.basename(current_file_path).split('.')[0]
    print(f"file name: {file_name}")

    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.Resize((config.data.image_size, config.data.image_size)),
        T.ToTensor()
    ])
    target_img = Image.open('images/mickey.png')
    target_img = transform(target_img)  # [0,1]

    if args.doc == "celeba":
        init_pattern = torch.randn((3, 64, 64), dtype=torch.float, device=device, requires_grad=True)
    else:
        init_pattern = torch.randn((3, 32, 32), dtype=torch.float, device=device, requires_grad=True)
    model_name = "blend"
    grad_scalar = 1
    lr = 1e-3
    similarity_weight = 1
    g_loss_weight = 5e-1
    x0_loss_weight = 5e-1

    # noise = torch.randn((eval_sample_n, 3, 32, 32), generator=torch.manual_seed(0))
    noise = torch.randn((args.batch, 3, 32, 32), generator=torch.manual_seed(0))
    noise = noise.to(device)
    row_number = int(math.sqrt(args.batch))
    optimizer = torch.optim.Adam([init_pattern], lr=lr, betas=(0.5, 0.9))

    timestep = 999
    from runners.diffusion_attack_d2i import get_beta_schedule
    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    sqrt_alpha_prod = alphas_cumprod[timestep] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()  # torch.size([1])
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timestep]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    grad_x0_xt = sqrt_alpha_prod.to(device)
    grad_xt_trigger = sqrt_one_minus_alpha_prod.to(device)

    # backdoor sampling
    alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)
    k_t = torch.randn_like(betas)
    for ii in range(config.diffusion.num_diffusion_timesteps):
        tmp_sum = torch.sqrt(1. - alphas_cumprod[ii])
        tmp_alphas = torch.flip(alphas[:ii + 1], [0])
        for jj in range(1, ii + 1):
            tmp_sum -= k_t[ii - jj] * torch.sqrt(torch.prod(tmp_alphas[:jj]))
        k_t[ii] = tmp_sum
    coef_miu = torch.sqrt(1. - alphas_cumprod_prev) * betas - (1. - alphas_cumprod_prev) * torch.sqrt(alphas) * k_t  

    input_path = f"{file_name}/{model_name}/generated_input"
    trigger_path = f"{file_name}/{model_name}/reverse_trigger"
    img_path = f"{file_name}/{model_name}/generated_img"
    for path in [input_path, trigger_path, img_path]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory {path} created!")

    # load the ckpt
    model = Model(config)
    if not args.use_pretrained:
        if getattr(config.sampling, "ckpt_id", None) is None:
            states = torch.load(os.path.join(args.log_path, "ckpt.pth"), map_location=config.device)
        else:
            states = torch.load(os.path.join(args.log_path, f"ckpt_{config.sampling.ckpt_id}.pth"), map_location=config.device)
        model = model.to(device)
        print("Loading model...")
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        if config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None
    model.eval()

    encoder, preprocess = clip.load("ViT-B/32", device=device)
    backdoor_detector = 0

    for epoch in range(100):  # begin inversion
        optimizer.zero_grad()
        pattern = 1. * init_pattern
        pattern = pattern.to(device)
        noise_plus_trigger = noise + pattern
        noise_plus_trigger.requires_grad_(True)

        n = config.sampling.batch_size
        x = torch.randn(n, config.data.channels, config.data.image_size, config.data.image_size,
                        device=config.device)

        runner.miu = init_pattern.to(config.device)
        miu = torch.stack([init_pattern.to(config.device)] * n)  # (batch,3,32,32)
        tmp_x = x.clone()
        x = args.gamma * x + miu  # N(miu,I)

        images = runner.sample_image_bd(x, model)
        print("images shape: ", images.shape)
        images = images.to(config.device)
        images.requires_grad_(True)

        images_new = transforms(images)
        # image_input = preprocess(images).unsqueeze(0)
        image_features = encoder.encode_image(images_new)

        flag, target_set, residual_set, loss_s = calc_loss_similarity(args, image_features)
        if flag:
            backdoor_detector = 1

        if residual_set != []:
            loss_s_weight = args.batch / len(residual_set)
        else:
            loss_s_weight = 1

        images_flattened = images.reshape(args.batch, -1)
        loss_x0_entropy_weight, loss_x0_e = calc_loss_entropy(target_set, residual_set, images_flattened)

        loss_x0 = loss_s_weight * loss_s + loss_x0_entropy_weight * loss_x0_e

        # print loss
        print(f"epoch: {epoch}")
        for loss, label in zip([loss_x0, loss_s, loss_x0_e], ["loss_x0", "loss_s", "loss_x0_e"]):
            if type(loss) == int or type(loss) == float:
                print(label, ": ", loss)
            else:
                print(label, ": ", loss.item())

        if type(loss_x0) != int and type(loss_x0) != float:
            loss_x0.backward()
            noise_plus_trigger.grad = grad_x0_xt * images.grad  # 16, 3, 32, 32
            pattern.grad = torch.mean(noise_plus_trigger.grad, dim=0, keepdim=False)  
            grad_initp_p = grad_scalar * element_wize_grad(init_pattern, pattern)
            init_pattern.grad = grad_initp_p * pattern.grad

            grad_loss_x0 = torch.mean(images.grad, dim=0, keepdim=False)  # 3, 32, 32
            grad_loss_x0 = grad_loss_x0.to(device)

            grad_p = grad_scalar * grad_loss_x0 * grad_x0_xt * grad_xt_trigger   # * grad_trigger_tanh

            init_pattern.grad = copy.deepcopy(grad_p)
            optimizer.step()

        asr = 0
        for i in target_set:
            if calc_mse(images_flattened[i], target_img) < 1e-3:
                asr += 1
        asr /= config.batch

        if epoch % 1 == 0:

            print(f"backdoor detector: {backdoor_detector}")
            print(f"Epoch {epoch} - images mean: {torch.mean(images)} - images var: {torch.var(images)}")
            images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
            image_grid = make_grid(images, rows=row_number, cols=row_number)
            image_grid.save(osp.join(img_path, f"img_epoch{epoch:04d}.png"))

            # gen_backdoor_target = ImagePathDataset(path=backdoor_path)[:].to(device)
            # ssim_sc = float(StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(images, target_np))


            #     f"- var loss: {var_loss.item():.4f} - norm loss: {norm_loss.item()}")
            # print(f"grad_p: {grad_p.cpu().numpy()}")

            # save trigger
            np.save(osp.join(trigger_path, f"trigger_epoch{epoch:04d}.npy"),
                    init_pattern.detach().cpu().numpy())
            trigger_np = init_pattern.permute(1, 2, 0).detach().cpu().numpy()
            print(f"Epoch {epoch} - trigger mean: {np.mean(trigger_np)} - trigger var: {np.var(trigger_np)}")
            trigger_np = (trigger_np * 255).round().astype("uint8")
            # print(pattern_np.shape)
            image = Image.fromarray(trigger_np)
            image.save(osp.join(trigger_path, f"trigger_epoch{epoch:04d}.png"))

            # inputs
            np.save(osp.join(input_path, f"input_epoch{epoch:04d}.npy"),
                    noise_plus_trigger.detach().cpu().numpy())
            noise_plus_trigger = noise_plus_trigger.permute(0, 2, 3, 1).detach().cpu().numpy()
            # print("noise ", noise_plus_trigger.shape)
            images = [Image.fromarray(noise) for noise in
                      np.squeeze((noise_plus_trigger * 255).round().astype("uint8"))]
            image_grid = make_grid(images, rows=row_number, cols=row_number)
            image_grid.save(osp.join(input_path, f"input_epoch{epoch:04d}.png"))

    return


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    runner = Diffusion(args, config)  # 这是din的diffusion
    trigger_inversion(args, config, runner)

    return 0


if __name__ == "__main__":
    sys.exit(main())
