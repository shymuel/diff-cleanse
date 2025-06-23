from dataclasses import dataclass
import argparse
import os
import json
from typing import Dict, Union
from torch.autograd import Variable
import torch
import tqdm as tqdm1
from dataset import DatasetLoader, Backdoor, ImagePathDataset
import numpy as np

from PIL import Image
from util import match_count
from fid_score import fid
from VillanDiffusion import Metric, update_score_file
from tqdm import tqdm


MODE_TRAIN: str = 'train'
MODE_RESUME: str = 'resume'
MODE_SAMPLING: str = 'sampling'
MODE_MEASURE: str = 'measure'
MODE_TRAIN_MEASURE: str = 'train+measure'
TASK_GENERATE: str = 'generate'
DEFAULT_PROJECT: str = "Default"
DEFAULT_SCHED: str = "DDPM-SCHED"
DEFAULT_BATCH: int = 512
DEFAULT_EVAL_MAX_BATCH: int = 1024
DEFAULT_EPOCH: int = 50
DEFAULT_INFER_STEPS: int = 1000
DEFAULT_DDIM_ETA: float = None
DEFAULT_LEARNING_RATE: float = None
DEFAULT_LEARNING_RATE_32: float = 2e-4
DEFAULT_LEARNING_RATE_256: float = 8e-5
DEFAULT_CLEAN_RATE: float = 1.0
DEFAULT_POISON_RATE: float = 0.007
DEFAULT_TRIGGER: str = Backdoor.TRIGGER_SM_BOX_MED
DEFAULT_TARGET: str = Backdoor.TARGET_CORNER
DEFAULT_DATASET_LOAD_MODE: str = DatasetLoader.MODE_FIXED
DEFAULT_GPU = '0'
DEFAULT_CKPT: str = None
DEFAULT_OVERWRITE: bool = False
DEFAULT_POSTFIX: str = ""
DEFAULT_FCLIP: str = 'o'
DEFAULT_SAVE_IMAGE_EPOCHS: int = 10
DEFAULT_SAVE_MODEL_EPOCHS: int = 5
DEFAULT_IS_SAVE_ALL_MODEL_EPOCHS: bool = False
DEFAULT_SAMPLE_EPOCH: int = None
DEFAULT_RESULT: int = '.'

NOT_MODE_TRAIN_OPTS = ['sample_ep']
NOT_MODE_TRAIN_MEASURE_OPTS = ['sample_ep']
MODE_RESUME_OPTS = ['project', 'mode', 'gpu', 'ckpt']
MODE_SAMPLING_OPTS = ['project', 'mode', 'eval_max_batch', 'gpu', 'fclip', 'ckpt', 'sample_ep']
MODE_MEASURE_OPTS = ['project', 'mode', 'eval_max_batch', 'gpu', 'fclip', 'ckpt', 'sample_ep', 'ddim',
                     'num_inference_steps']
# IGNORE_ARGS = ['overwrite']
IGNORE_ARGS = ['overwrite', 'is_save_all_model_epochs']


def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--project', '-pj', required=False, type=str, default= "Reverse", help='Project name')
    parser.add_argument('--mode', '-m', required=False, type=str, help='Train or test the model', default=MODE_MEASURE,
                        choices=[MODE_TRAIN, MODE_MEASURE, MODE_RESUME, MODE_SAMPLING, MODE_MEASURE, MODE_TRAIN_MEASURE])
    parser.add_argument('--dataset', '-ds', type=str, help='Training dataset',
                        choices=[DatasetLoader.MNIST, DatasetLoader.CIFAR10, DatasetLoader.CELEBA,
                                 DatasetLoader.CELEBA_HQ])
    parser.add_argument('--batch', '-b', type=int, help=f"Batch size, default for train: {DEFAULT_BATCH}")
    parser.add_argument('--eval_max_batch', '-eb', type=int,
                        help=f"Batch size of sampling, default for train: {DEFAULT_EVAL_MAX_BATCH}")
    parser.add_argument('--epoch', '-e', type=int, help=f"Epoch num, default for train: {DEFAULT_EPOCH}")
    parser.add_argument('--learning_rate', '-lr', type=float,
                        help=f"Learning rate, default for 32 * 32 image: {DEFAULT_LEARNING_RATE_32}, default for larger images: {DEFAULT_LEARNING_RATE_256}")
    parser.add_argument('--clean_rate', '-cr', type=float, help=f"Clean rate, default for train: {DEFAULT_CLEAN_RATE}")
    parser.add_argument('--poison_rate', '-pr', type=float,
                        help=f"Poison rate, default for train: {DEFAULT_POISON_RATE}")
    parser.add_argument('--clip_norm', default= 0.01, type=int, help="Norm for clipping.")
    parser.add_argument('--trigger', '-tr', type=str, help=f"Trigger pattern, default for train: {DEFAULT_TRIGGER}")
    parser.add_argument('--target', '-ta', type=str, help=f"Target pattern, default for train: {DEFAULT_TARGET}")
    parser.add_argument('--dataset_load_mode', '-dlm', type=str,
                        help=f"Mode of loading dataset, default for train: {DEFAULT_DATASET_LOAD_MODE}",
                        choices=[DatasetLoader.MODE_FIXED, DatasetLoader.MODE_FLEX])
    parser.add_argument('--gpu', '-g', type=str, help=f"GPU usage, default for train/resume: {DEFAULT_GPU}")
    parser.add_argument('--ckpt', '-c', type=str, help=f"Load from the checkpoint, default: {DEFAULT_CKPT}")
    parser.add_argument('--overwrite', '-o', action='store_true',
                        help=f"Overwrite the existed training result or not, default for train/resume: {DEFAULT_CKPT}")
    parser.add_argument('--postfix', '-p', type=str,
                        help=f"Postfix of the name of the result folder, default for train/resume: {DEFAULT_POSTFIX}")
    parser.add_argument('--fclip', '-fc', type=str,
                        help=f"Force to clip in each step or not during sampling/measure, default for train/resume: {DEFAULT_FCLIP}",
                        choices=['w', 'o'])
    parser.add_argument('--save_image_epochs', '-sie', type=int,
                        help=f"Save sampled image per epochs, default: {DEFAULT_SAVE_IMAGE_EPOCHS}")
    parser.add_argument('--save_model_epochs', '-sme', type=int,
                        help=f"Save model per epochs, default: {DEFAULT_SAVE_MODEL_EPOCHS}")
    parser.add_argument('--is_save_all_model_epochs', '-isame', action='store_true', help=f"")
    parser.add_argument('--sample_ep', '-se', type=int,
                        help=f"Select i-th epoch to sample/measure, if no specify, use the lastest saved model, default: {DEFAULT_SAMPLE_EPOCH}")
    parser.add_argument('--result', '-res', type=str, help=f"Output file path, default: {DEFAULT_RESULT}")
    parser.add_argument('--num_inference_steps', default=50, type=int, help="Number of time steps.")
    parser.add_argument('--ddim', action='store_true', help="Whether to adopt the ddim sampling")

    # hyperparmeter for reverse engineering
    parser.add_argument('--weight_decay', type=float, default=5e-5, help="Trade-off coefficient.")
    parser.add_argument('--lr', type=float, default=0.5, help="Learning rate for reversed engineering")
    parser.add_argument("--iteration", type=int, default=3000, help="Iterations for Trigger Estimation")
    parser.add_argument("--out_dir", type=str, default="./log/log",
                        help="Path to the config file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch_size of reverse engineer")
    parser.add_argument('--num_steps', default=1000, type=int, help="Number of steps for ScoreVde sampler.")
    # hyperparmeter for reverse engineering


    args = parser.parse_args()
    return args


@dataclass
class TrainingConfig:
    project: str = DEFAULT_PROJECT
    task: str = TASK_GENERATE
    batch: int = DEFAULT_BATCH
    sched: str = DEFAULT_SCHED
    infer_steps: int = DEFAULT_INFER_STEPS
    epoch: int = DEFAULT_EPOCH
    ddim_eta: float = DEFAULT_DDIM_ETA
    eval_max_batch: int = DEFAULT_EVAL_MAX_BATCH
    learning_rate: float = DEFAULT_LEARNING_RATE
    clean_rate: float = DEFAULT_CLEAN_RATE
    poison_rate: float = DEFAULT_POISON_RATE
    trigger: str = DEFAULT_TRIGGER
    target: str = DEFAULT_TARGET
    dataset_load_mode: str = DEFAULT_DATASET_LOAD_MODE
    gpu: str = DEFAULT_GPU
    ckpt: str = DEFAULT_CKPT
    overwrite: bool = DEFAULT_OVERWRITE
    postfix: str = DEFAULT_POSTFIX
    fclip: str = DEFAULT_FCLIP
    save_image_epochs: int = DEFAULT_SAVE_IMAGE_EPOCHS
    save_model_epochs: int = DEFAULT_SAVE_MODEL_EPOCHS
    is_save_all_model_epochs: bool = DEFAULT_IS_SAVE_ALL_MODEL_EPOCHS
    sample_ep: int = DEFAULT_SAMPLE_EPOCH
    result: str = DEFAULT_RESULT

    eval_sample_n: int = 16  # how many images to sample during evaluation
    measure_sample_n: int = 16
    batch_32: int = 128
    batch_256: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate_32_scratch: float = 2e-4
    learning_rate_256_scratch: float = 2e-5
    lr_warmup_steps: int = 500
    # save_image_epochs: int = 1
    mixed_precision: str = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    # mixed_precision: str = 'no'  # `no` for float32, `fp16` for automatic mixed precision

    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 1234
    dataset_path: str = 'datasets'
    ckpt_dir: str = 'ckpt'
    data_ckpt_dir: str = 'data.ckpt'
    ep_model_dir: str = 'epochs'
    ckpt_path: str = None
    data_ckpt_path: str = None


def naming_fn(config: TrainingConfig):
    add_on: str = ""
    # add_on += "_clip" if config.clip else ""
    add_on += f"_{config.postfix}" if config.postfix else ""
    return f'res_{config.ckpt}_{config.dataset}_ep{config.epoch}_c{config.clean_rate}_p{config.poison_rate}_{config.trigger}-{config.target}{add_on}'


def read_json(args: argparse.Namespace, file: str):
    with open(os.path.join(args.ckpt, file), "r") as f:
        return json.load(f)


def write_json(content: Dict, config: argparse.Namespace, file: str):
    with open(os.path.join(config.output_dir, file), "w") as f:
        return json.dump(content, f, indent=2)


def setup():
    args_file: str = "args.json"
    config_file: str = "config.json"
    sampling_file: str = "sampling.json"
    measure_file: str = "measure.json"

    args: argparse.Namespace = parse_args()
    config: TrainingConfig = TrainingConfig()
    args_data: Dict = {}

    if args.mode == MODE_RESUME or args.mode == MODE_SAMPLING or args.mode == MODE_MEASURE:
        with open(os.path.join(args.ckpt, args_file), "r") as f:
            args_data = json.load(f)

        for key, value in args_data.items():
            if value != None:
                setattr(config, key, value)
        setattr(config, "output_dir", args.ckpt)

    for key, value in args.__dict__.items():
        if args.mode == MODE_TRAIN and (key not in NOT_MODE_TRAIN_OPTS) and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_TRAIN_MEASURE and (key not in NOT_MODE_TRAIN_MEASURE_OPTS) and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_RESUME and key in MODE_RESUME_OPTS and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_SAMPLING and key in MODE_SAMPLING_OPTS and value != None:
            setattr(config, key, value)
        elif args.mode == MODE_MEASURE and key in MODE_MEASURE_OPTS and value != None:
            setattr(config, key, value)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.gpu)

    print(f"PyTorch detected number of available devices: {torch.cuda.device_count()}")
    setattr(config, "device_ids", [int(i) for i in range(len(config.gpu.split(',')))])

    # sample_ep options
    if isinstance(config.sample_ep, int):
        if config.sample_ep < 0:
            config.sample_ep = None

    # Clip option
    if config.fclip == 'w':
        setattr(config, "clip", True)
    elif config.fclip == 'o':
        setattr(config, "clip", False)
    else:
        setattr(config, "clip", None)

    # Determine gradient accumulation & Learning Rate
    bs = 0
    if config.dataset in [DatasetLoader.CIFAR10, DatasetLoader.MNIST]:
        bs = config.batch_32
        if config.learning_rate == None:
            if config.ckpt == None:
                config.learning_rate = config.learning_rate_32_scratch
            else:
                config.learning_rate = DEFAULT_LEARNING_RATE_32
    elif config.dataset in [DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ, DatasetLoader.LSUN_CHURCH,
                            DatasetLoader.LSUN_BEDROOM]:
        bs = config.batch_256
        if config.learning_rate == None:
            if config.ckpt == None:
                config.learning_rate = config.learning_rate_256_scratch
            else:
                config.learning_rate = DEFAULT_LEARNING_RATE_256
    else:
        raise NotImplementedError()
    if bs % config.batch != 0:
        raise ValueError(f"batch size {config.batch} should be divisible to {bs} for dataset {config.dataset}")
    if bs < config.batch:
        raise ValueError(f"batch size {config.batch} should be smaller or equal to {bs} for dataset {config.dataset}")
    config.gradient_accumulation_steps = int(bs // config.batch)

    if args.mode == MODE_TRAIN or args.mode == MODE_TRAIN_MEASURE:
        setattr(config, "output_dir", os.path.join(config.result, naming_fn(config=config)))

    print(f"MODE: {config.mode}")
    if config.mode == MODE_TRAIN or args.mode == MODE_TRAIN_MEASURE:
        if not config.overwrite and os.path.isdir(config.output_dir):
            raise ValueError(
                f"Output directory: {config.output_dir} has already been created, please set overwrite flag --overwrite or -o")

        os.makedirs(config.output_dir, exist_ok=True)

        write_json(content=vars(args), config=config, file=args_file)
        write_json(content=config.__dict__, config=config, file=config_file)
    elif config.mode == MODE_SAMPLING:
        write_json(content=config.__dict__, config=config, file=sampling_file)
    elif config.mode == MODE_MEASURE or args.mode == MODE_TRAIN_MEASURE:
        write_json(content=config.__dict__, config=config, file=measure_file)
    elif config.mode == MODE_RESUME:
        pass
    else:
        raise NotImplementedError(f"Mode: {config.mode} isn't defined")

    if config.ckpt_path == None:
        config.ckpt_path = os.path.join(config.output_dir, config.ckpt_dir)
        config.data_ckpt_path = os.path.join(config.output_dir, config.data_ckpt_dir)
        os.makedirs(config.ckpt_path, exist_ok=True)

    print(f"Argument Final: {config.__dict__}")
    return config, args


config, args = setup()
"""## Config

For convenience, we define a configuration grouping all the training hyperparameters. This would be similar to the arguments used for a [training script](https://github.com/huggingface/diffusers/tree/main/examples).
Here we choose reasonable defaults for hyperparameters like `num_epochs`, `learning_rate`, `lr_warmup_steps`, but feel free to adjust them if you train on your own dataset. For example, `num_epochs` can be increased to 100 for better visual quality.
"""

from torch import nn
from accelerate import Accelerator
# from diffusers.hub_utils import init_git_repo

from diffusers import DDIMScheduler, DDPMPipeline, ScoreSdeVePipeline, ScoreSdeVeScheduler
from reverse_pipeline import DDIMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from model import DiffuserModelSched
from reverse_loss import p_losses_diffuser


def get_accelerator(config: TrainingConfig):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # log_with=["tensorboard"],
        # logging_dir=os.path.join(config.output_dir, "logs")
    )
    return accelerator


def init_tracker(config: TrainingConfig, accelerator: Accelerator):
    tracked_config = {}
    for key, val in config.__dict__.items():
        if isinstance(val, int) or isinstance(val, float) or isinstance(val, str) or isinstance(val,
                                                                                                bool) or isinstance(val,
                                                                                                                    torch.Tensor):
            tracked_config[key] = val
    accelerator.init_trackers(config.project, config=tracked_config)


def get_data_loader(config: TrainingConfig):
    ds_root = os.path.join(config.dataset_path)
    dsl = DatasetLoader(root=ds_root, name=config.dataset, batch_size=config.batch).set_poison(
        trigger_type=config.trigger, target_type=config.target, clean_rate=config.clean_rate,
        poison_rate=config.poison_rate).prepare_dataset(mode=config.dataset_load_mode)
    print(f"datasetloader len: {len(dsl)}")
    return dsl


def get_repo(config: TrainingConfig, accelerator: Accelerator):
    repo = None
    # if accelerator.is_main_process:
    #     if config.push_to_hub:
    #         repo = init_git_repo(config, at_init=True)
    #     init_tracker(config=config, accelerator=accelerator)
    return repo


def get_model_optim_sched(config: TrainingConfig, accelerator: Accelerator, dataset_loader: DatasetLoader):
    if config.ckpt != None:
        if config.sample_ep != None and config.mode in [MODE_MEASURE, MODE_SAMPLING]:
            ep_model_path = get_ep_model_path(config=config, dir=config.ckpt, epoch=config.sample_ep)
            model, noise_sched = DiffuserModelSched.get_pretrained(ckpt=ep_model_path, clip_sample=config.clip)
        else:
            model, noise_sched = DiffuserModelSched.get_pretrained(ckpt=config.ckpt, clip_sample=config.clip)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        model, noise_sched = DiffuserModelSched.get_model_sched(image_size=dataset_loader.image_size,
                                                                channels=dataset_loader.channel,
                                                                model_type=DiffuserModelSched.MODEL_DEFAULT,
                                                                clip_sample=config.clip)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    model = nn.DataParallel(model, device_ids=config.device_ids)
    lr_sched = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(dataset_loader.num_batch * config.epoch),
    )
    cur_epoch = cur_step = 0
    accelerator.register_for_checkpointing(model, optimizer, lr_sched)
    if config.mode == MODE_RESUME:
        if config.ckpt == None:
            raise ValueError(f"Argument 'ckpt' shouldn't be None for resume mode")
        accelerator.load_state(config.ckpt_path)
        data_ckpt = torch.load(config.data_ckpt_path)
        cur_epoch = data_ckpt['epoch']
        cur_step = data_ckpt['step']
    return model, optimizer, lr_sched, noise_sched, cur_epoch, cur_step


def measure_subfolder_naming_ext(config: TrainingConfig):
    res = ""
    if config.sched != None:
        res += f"_{config.sched}-{config.infer_steps}" if config.sched != None else ""
    res += f"_{config.measure_sample_n}"
    return res

def init_train(config: TrainingConfig, dataset_loader: DatasetLoader):
    # Initialize accelerator and tensorboard logging
    accelerator = get_accelerator(config=config)
    repo = get_repo(config=config, accelerator=accelerator)
    model, optimizer, lr_sched, noise_sched, cur_epoch, cur_step = get_model_optim_sched(config=config,
                                                                                         accelerator=accelerator,
                                                                                         dataset_loader=dataset_loader)
    dataloader = dataset_loader.get_dataloader()
    model, optimizer, dataloader, lr_sched = accelerator.prepare(
        model, optimizer, dataloader, lr_sched
    )
    return accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step



def reverse_ncsn(model, dataset_loader, ncsn_noise_sched, config: TrainingConfig, folder_name: Union[int, str], pipeline,
            args=None):
    folder_path_ls = [config.output_dir, folder_name]
    if config.sample_ep != None:
        folder_path_ls += [f"ep{config.sample_ep}"]
    mu = Variable(
        -torch.rand(pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size).cuda(),
        requires_grad=True)  # mu是随机生成的trigger初始值
    optim = torch.optim.SGD([mu], lr=args.lr, weight_decay=0)
    iterations = args.iteration
    batch_size = args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, iterations)
    args.out_dir2 = args.out_dir + args.project + "/log_" + str(args.weight_decay) + "_" + str(
        args.num_steps) + "_" + str(
        args.iteration) + "_" + str(args.batch_size) + "_" + str(args.lr)  + "/"
    os.makedirs(args.out_dir2, exist_ok=True)
    model.eval()

    mu_path = os.path.join(args.out_dir2, "reverse.pkl")
    print(f"mu path: {mu_path}")
    if os.path.exists(mu_path):  # do not double calculate mu
        return
    for _ in tqdm1.tqdm(
            range(args.iteration), desc="Trigger Estimation"
    ):
        #################################################
        #       Reversed loss for Trigger Estimation    #
        #################################################
        bs = batch_size
        timesteps = torch.randint(noise_sched.num_train_timesteps - 10, noise_sched.num_train_timesteps,
                                  (bs,)).long().cuda()
        fake_image = torch.randn(
            (batch_size, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size)).cuda()

        loss = p_losses_diffuser(ncsn_noise_sched, model=model, x_start=fake_image, R=mu, timesteps=timesteps, sde_type="SDE-VE")
        loss_update = loss - args.weight_decay * torch.norm(mu, p=1)
        optim.zero_grad()
        loss_update.backward()
        optim.step()
        scheduler.step()
        torch.save({"mu": mu}, mu_path)
        print(torch.flatten(mu.detach()))


    optim = torch.optim.SGD([mu], lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, iterations // 3)

    for i in tqdm1.tqdm(
            range(iterations, int(iterations * 4 / 3)), desc="Trigger Refinement"
    ):
        n = batch_size
        noise = torch.randn((batch_size, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size)).cuda()
        batch_miu = torch.stack([mu.cuda()] * n)  # (batch,3,32,32)
        x = noise + batch_miu
        #################################
        #      Generate  image          #
        #################################
        with torch.no_grad():
            generate_image = pipeline(batch_size=args.batch_size, generator=None,
                init=x, output_type=None, num_inference_steps=args.num_steps).images

        # save the generated image to check
        images = [Image.fromarray(image) for image in np.squeeze((generate_image * 255).round().astype("uint8"))]
        for j, img in enumerate(tqdm(images)):
            img.save(os.path.join("D:\\TERD\\BadDiffusion\\ncsn_test", f"refine_{i}_{j}.png"))
        del images

        # print(pipeline)
        # print(pipeline.scheduler)
        # change the timestep and sigmas manually
        pipeline.scheduler.set_timesteps(2000)
        pipeline.scheduler.set_sigmas(2000)
        generate_image = torch.from_numpy(generate_image)
        generate_image = generate_image.permute(0, 3, 1, 2).to("cuda:0")  # If you want to move it back to GPU, otherwise skip this step
        print("generate image shape: ", generate_image.shape)
        #################################################
        #       Reversed loss for trigger refinement    #
        #################################################

        bs = generate_image.shape[0]
        timesteps = torch.randint(ncsn_noise_sched.num_train_timesteps - 10, ncsn_noise_sched.num_train_timesteps,
                                  (bs,)).long().cuda()
        loss_1 = p_losses_diffuser(ncsn_noise_sched, model=model, x_start=generate_image, R=mu, timesteps=timesteps, last=True, sde_type="SDE-VE")
        print(f"loss_1: {loss_1.item()}")
        timesteps = torch.randint(0, 10, (bs,)).long().cuda()
        loss_2 = p_losses_diffuser(ncsn_noise_sched, model=model, x_start=generate_image, R=mu, timesteps=timesteps, last=False, sde_type="SDE-VE")
        loss_update = (loss_1+loss_2)/2 - args.weight_decay * torch.norm(mu, p=1)
        print(f"loss_1: {loss_1.item()}, loss_2: {loss_2.item()}, loss_update: {loss_update.item()}, mu norm: {args.weight_decay * torch.norm(mu, p=1)}")
        # print(f"loss update: {loss_update.item()}")
        optim.zero_grad()
        loss_update.backward()
        torch.nn.utils.clip_grad_norm_([mu], args.clip_norm)
        optim.step()
        scheduler.step()
        torch.save({"mu": mu}, mu_path)
        print(torch.flatten(mu.detach()))


def measure(config: TrainingConfig, dataset_loader: DatasetLoader,
            folder_name: Union[int, str], pipeline, num_backdoor=100, resample: bool = False, recomp: bool = True):
    # 这个measure的路径还有问题
            # num_clean = 16, num_backdoor = 10000, resample: bool = False, recomp: bool = True):
    from tqdm import tqdm
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
                generator=torch.manual_seed(config.seed + i))
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

    args.out_dir2 = args.out_dir + args.project + "/log_" + str(args.weight_decay) + "_" + str(
        args.num_steps) + "_" + str(
        args.iteration) + "_" + str(args.batch_size) + "_" + str(args.lr) + "/"
    os.makedirs(args.out_dir2, exist_ok=True)
    model.eval()

    mu_path = os.path.join(args.out_dir2, "reverse.pkl")
    inverted_trigger = torch.load(mu_path)
    inverted_trigger = inverted_trigger["mu"]  # a tensor of shape [3,32,32]

    score_file = os.path.join(args.out_dir, "score.json")
    fid_sc = mse_sc = ssim_sc = lpips_sc = 0.0
    re_comp_clean_metric = False
    re_comp_backdoor_metric = False

    # Random Number Generator
    rng = torch.Generator()
    rng.manual_seed(config.seed)

    # Dataset samples
    step = dataset_loader.num_batch * (config.sample_ep + 1 if config.sample_ep != None else config.epoch)

    # Folders
    if config.dataset == "CIFAR10":
        dataset_img_dir = "E:\\00_dataset\\cifar10.npz"
        config.eval_max_batch = 16
    else:
        dataset_img_dir = "E:\\00_dataset\\celebahq256.npz"  # 这个还没准备
        config.eval_max_batch = 2
    folder_path_ls = [args.out_dir, folder_name]
    if config.sample_ep != None:
        folder_path_ls += [f"ep{config.sample_ep}"]

    backdoor_folder = "backdoor" + ("_noclip" if not config.clip else "") + measure_subfolder_naming_ext(config=config)
    backdoor_path = os.path.join(*folder_path_ls, backdoor_folder)
    print(f"backdoor path: {backdoor_path}")
    # Sampling
    if not os.path.isdir(backdoor_path) or match_count(dir=backdoor_path) < num_backdoor or resample:
        print("num backdoor: ", num_backdoor)
        batch_sampling_save(sample_n=num_backdoor, num_inference_steps=config.infer_steps, ddim_eta=config.ddim_eta,
                            pipeline=pipeline,
                            path=backdoor_path, trigger=inverted_trigger.unsqueeze(0),
                            max_batch_n=config.eval_max_batch, rng=rng)
        re_comp_backdoor_metric = True

    # print("Donot compute metrics! ")
    # return

    # Compute Score
    if re_comp_backdoor_metric or recomp:
        print("backdoor measure")
        device = torch.device(config.device_ids[0])
        # device = "cpu"
        # gen_backdoor_target = torch.from_numpy(backdoor_sample_imgs)
        # print(f"backdoor_sample_imgs shape: {backdoor_sample_imgs.shape}")
        gen_backdoor_target = ImagePathDataset(path=backdoor_path, njobs=1)[:].to(device)

        reps = ([len(gen_backdoor_target)] + ([1] * (len(dsl.target.shape))))
        if config.sde_type == DiffuserModelSched.SDE_VE:
            backdoor_target = torch.squeeze((dsl.target.repeat(*reps)).clamp(0, 1)).to(device)
        else:
            backdoor_target = torch.squeeze((dsl.target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(device)
        # backdoor_target = torch.squeeze((dsl.target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(device)

        print(
            f"gen_backdoor_target: {gen_backdoor_target.shape}, vmax: {torch.max(gen_backdoor_target)}, vmin: {torch.min(backdoor_target)} | backdoor_target: {backdoor_target.shape}, vmax: {torch.max(backdoor_target)}, vmin: {torch.min(backdoor_target)}")
        # mse_sc = float(nn.MSELoss(reduction='mean')(gen_backdoor_target, backdoor_target))
        # ssim_sc = float(StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(gen_backdoor_target, backdoor_target))
        mse_sc = Metric.mse_batch(a=gen_backdoor_target, b=backdoor_target, max_batch_n=config.eval_max_batch)
        ssim_sc = Metric.ssim_batch(a=gen_backdoor_target, b=backdoor_target, max_batch_n=config.eval_max_batch,
                                    device=device)
        asr_sc = Metric.mse_thres_batch(a=gen_backdoor_target, b=backdoor_target,
                                        thres=0.01, max_batch_n=config.eval_max_batch)
    print(f"[{config.sample_ep}] FID: {fid_sc}, MSE: {mse_sc}, SSIM: {ssim_sc}, ASR: {asr_sc}")
    sc = update_score_file(config=config, score_file=score_file, fid_sc=fid_sc, mse_sc=mse_sc, ssim_sc=ssim_sc, asr_sc=asr_sc)



def get_ep_model_path(config: TrainingConfig, dir: Union[str, os.PathLike], epoch: int):
    return os.path.join(dir, config.ep_model_dir, f"ep{epoch}")

"""
Let's reverse the trigger!
"""
dsl = get_data_loader(config=config)
accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step = init_train(config=config, dataset_loader=dsl)
# ncsn_noise_sched = noise_sched

ncsn_noise_sched = ScoreSdeVeScheduler(num_train_timesteps=2000, sigma_min=0.01, sigma_max=380.0, sampling_eps=1e-05, correct_steps=1, snr=0.075)
print(f"ncsn noise sched: {ncsn_noise_sched}")
pipeline = ScoreSdeVePipeline(unet=accelerator.unwrap_model(model), scheduler=ncsn_noise_sched)
reverse_ncsn(config=config, dataset_loader=dsl, ncsn_noise_sched=ncsn_noise_sched, model=model, folder_name='measure',
            pipeline=pipeline, args=args)
pipeline = ScoreSdeVePipeline(unet=accelerator.unwrap_model(model), scheduler=ncsn_noise_sched)
measure(config=config, dataset_loader=dsl, folder_name='measure',
            pipeline=pipeline)
accelerator.end_training()

# 100个epoch，trigger estimation: 10min, trigger refinement: 26min
