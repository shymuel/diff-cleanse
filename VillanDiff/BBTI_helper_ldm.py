import copy
from dataclasses import dataclass
import argparse
import os
import json
import traceback
from typing import Dict, Union
import warnings

import torch
import math
from dataset import DatasetLoader, Backdoor, ImagePathDataset
from util import Log
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import os.path as osp

from torchvision import transforms
from PIL import Image

MODE_SAMPLING: str = 'sampling'
MODE_MEASURE: str = 'measure'
MODE_REVERSE: str = 'reverse'

DEFAULT_PROJECT: str = "Default"
DEFAULT_BATCH: int = 16
DEFAULT_EVAL_MAX_BATCH: int = 256
DEFAULT_EPOCH: int = 50
DEFAULT_LEARNING_RATE: float = None
DEFAULT_LEARNING_RATE_32: float = 2e-4
DEFAULT_LEARNING_RATE_256: float = 8e-5
DEFAULT_CLEAN_RATE: float = 1.0
DEFAULT_POISON_RATE: float = 0.007
DEFAULT_TRIGGER: str = Backdoor.TRIGGER_SM_BOX_MED  # grey box with size=14*14
DEFAULT_TARGET: str = Backdoor.TARGET_BOX
DEFAULT_DATASET_LOAD_MODE: str = DatasetLoader.MODE_FIXED
DEFAULT_GPU = '0'
DEFAULT_SAVE_EPOCH = 10
DEFAULT_CKPT: str = None
DEFAULT_OVERWRITE: bool = False
DEFAULT_POSTFIX: str = ""
DEFAULT_FCLIP: str = 'o'
DEFAULT_SAVE_IMAGE_EPOCHS: int = 20
DEFAULT_SAVE_MODEL_EPOCHS: int = 5
DEFAULT_IS_SAVE_ALL_MODEL_EPOCHS: bool = False
DEFAULT_SAMPLE_EPOCH: int = None
DEFAULT_RESULT: int = '.'

# IGNORE_ARGS = ['overwrite']
IGNORE_ARGS = ['overwrite', 'is_save_all_model_epochs']


def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--project', '-pj', required=False, type=str, help='Project name')
    parser.add_argument('--mode', '-m', required=True, type=str, help='Train or test the model',
                        choices=[MODE_REVERSE])
    parser.add_argument('--dataset', '-ds', type=str, help='Training dataset',
                        choices=[DatasetLoader.MNIST, DatasetLoader.CIFAR10,
                                 DatasetLoader.CELEBA, DatasetLoader.CELEBA_HQ])
    parser.add_argument('--batch', '-b', type=int, help=f"Batch size, default for train: {DEFAULT_BATCH}")
    parser.add_argument('--sched', '-sc', type=str, help='Noise scheduler',
                        choices=["DDPM-SCHED", "DDIM-SCHED", "DPM_SOLVER_PP_O1-SCHED",
                                 "DPM_SOLVER_O1-SCHED", "DPM_SOLVER_PP_O2-SCHED", "DPM_SOLVER_O2-SCHED",
                                 "DPM_SOLVER_PP_O3-SCHED", "DPM_SOLVER_O3-SCHED", "UNIPC-SCHED", "PNDM-SCHED",
                                 "DEIS-SCHED", "HEUN-SCHED", "SCORE-SDE-VE-SCHED"])
    parser.add_argument('--infer_steps', '-is', type=int, help='Number of inference steps')
    parser.add_argument('--eval_max_batch', '-eb', type=int,
                        help=f"Batch size of sampling, default for train: {DEFAULT_EVAL_MAX_BATCH}")  # 256
    parser.add_argument('--epoch', '-e', type=int, help=f"Epoch num, default for train: {DEFAULT_EPOCH}")
    parser.add_argument('--learning_rate', '-lr', type=float,
                        help=f"Learning rate, default for 32 * 32 image: {DEFAULT_LEARNING_RATE_32}, default for larger images: {DEFAULT_LEARNING_RATE_256}")
    parser.add_argument('--clean_rate', '-cr', type=float, help=f"Clean rate, default for train: {DEFAULT_CLEAN_RATE}")
    parser.add_argument('--poison_rate', '-pr', type=float,
                        help=f"Poison rate, default for train: {DEFAULT_POISON_RATE}")
    parser.add_argument('--trigger', '-tr', type=str, help=f"Trigger trigger, default for train: {DEFAULT_TRIGGER}")
    parser.add_argument('--target', '-ta', type=str, help=f"Target trigger, default for train: {DEFAULT_TARGET}")
    parser.add_argument('--dataset_load_mode', '-dlm', type=str,
                        help=f"Mode of loading dataset, default for train: {DEFAULT_DATASET_LOAD_MODE}",
                        choices=[DatasetLoader.MODE_FIXED, DatasetLoader.MODE_FLEX])
    parser.add_argument('--gpu', '-g', type=str, help=f"GPU usage, default for train/resume: {DEFAULT_GPU}")
    parser.add_argument('--save_epoch', type=int, help=f"save the trigger every {DEFAULT_SAVE_EPOCH} epoches.",
                        default=10)
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
    args = parser.parse_args()
    return args


@dataclass
class TrainingConfig:
    project: str = DEFAULT_PROJECT
    batch: int = DEFAULT_BATCH
    epoch: int = DEFAULT_EPOCH
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

    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 0
    dataset_path: str = 'datasets'
    ckpt_dir: str = 'ckpt'
    data_ckpt_dir: str = 'data.ckpt'
    ep_model_dir: str = 'epochs'
    ckpt_path: str = None
    data_ckpt_path: str = None
    # hub_token = "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"


def naming_fn(config: TrainingConfig):  # create a string based on the config, which is the name of the model
    add_on: str = ""
    # add_on += "_clip" if config.clip else ""
    add_on += f"_{config.postfix}" if config.postfix else ""
    return f'res_{config.ckpt}_{config.dataset}_ep{config.epoch}_c{config.clean_rate}_p{config.poison_rate}_{config.trigger}-{config.target}{add_on}'


def read_json(args: argparse.Namespace, file: str):  # read the checkpoint
    with open(os.path.join(args.ckpt, file), "r") as f:
        return json.load(f)


def write_json(content: Dict, config: argparse.Namespace, file: str):  # save the checkpoint
    with open(os.path.join(config.output_dir, file), "w") as f:
        return json.dump(content, f, indent=2)


def setup():
    args_file: str = "args.json"
    config_file: str = "config.json"
    sampling_file: str = "sampling.json"
    measure_file: str = "measure.json"

    args: argparse.Namespace = parse_args()  # args is an instance of 'argparse.Namespace'
    config: TrainingConfig = TrainingConfig()
    args_data: Dict = {}

    if args.mode == MODE_REVERSE:
        model_path = "H:\\villandiff_models"
        args_path = os.path.join(model_path, args.ckpt)  # 加载ckpt
        with open(os.path.join(args_path, args_file), "r") as f:
            args_data = json.load(f)

        for key, value in args_data.items():
            if value != None:
                setattr(config, key, value)
        setattr(config, "output_dir", args.ckpt)

    for key, value in args.__dict__.items():
        if args.mode == MODE_REVERSE and value != None:
            setattr(config, key, value)
        elif value != None and not (key in IGNORE_ARGS):
            raise NotImplementedError(f"Argument: {key}={value} isn't used in mode: {args.mode}")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.gpu)

    print(f"PyTorch detected number of availabel devices: {torch.cuda.device_count()}")
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
    if config.dataset in [DatasetLoader.CIFAR10, DatasetLoader.MNIST, DatasetLoader.CELEBA_HQ_LATENT_PR05,
                          DatasetLoader.CELEBA_HQ_LATENT]:
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

    print(f"target: {config.target}")
    print(f"MODE: {config.mode}")

    if config.mode == MODE_REVERSE:
        pass
    else:
        raise NotImplementedError(f"Mode: {config.mode} isn't defined")

    if config.ckpt_path == None:
        config.ckpt_path = os.path.join(config.output_dir, config.ckpt_dir)
        config.data_ckpt_path = os.path.join(config.output_dir, config.data_ckpt_dir)
        os.makedirs(config.ckpt_path, exist_ok=True)

    name_id = str(config.output_dir).split('/')[-1]
    print(f"Argument Final: {config.__dict__}")
    return config


import numpy as np
from PIL import Image
from torch import nn
from accelerate import Accelerator
# from diffusers.hub_utils import init_git_repo, push_to_hub
from tqdm.auto import tqdm

from diffusers import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

from model import DiffuserModelSched, batch_sampling, batch_sampling_save
from util import Samples, MemoryLog


def get_accelerator(config: TrainingConfig):  # to accelerate the train process
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,  # open or close mixed-precision training
        # mixed_precision=TrainingConfig.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,  # set steps for gradient accumulation
    )
    return accelerator


def init_tracker(config: TrainingConfig, accelerator: Accelerator):
    '''set parameters to be recorded.'''
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
        trigger_type=config.trigger,
        target_type=config.target, clean_rate=config.clean_rate, poison_rate=config.poison_rate).prepare_dataset(
        mode=config.dataset_load_mode)
    # image_size = dsl.image_size
    # channels = dsl.channel
    # dataset = dsl.get_dataset()
    # loader = dsl.get_dataloader()
    print(f"datasetloader len: {len(dsl)}")
    return dsl


def get_repo(config: TrainingConfig, accelerator: Accelerator):
    '''useless'''
    repo = None
    if accelerator.is_main_process:
        # if config.push_to_hub:
        #     repo = init_git_repo(config, at_init=True)
        # accelerator.init_trackers(config.output_dir, config=config.__dict__)
        init_tracker(config=config, accelerator=accelerator)
    return repo


def get_model_optim_sched(config: TrainingConfig, accelerator: Accelerator, dataset_loader: DatasetLoader):
    if config.ckpt != None:
        if config.sample_ep != None and config.mode in [MODE_MEASURE, MODE_SAMPLING]:
            ep_model_path = get_ep_model_path(config=config, dir=config.ckpt, epoch=config.sample_ep)
            model, vae, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=ep_model_path,
                                                                                      noise_sched_type=config.sched,
                                                                                      clip_sample=config.clip)
        # else:
        #     model, noise_sched = DiffuserModelSched.get_pretrained(ckpt=config.ckpt, clip_sample=config.clip)
        #     warnings.warn(Log.warning(f"No such pretrained model: {ep_model_path}, load from ckpt: {config.ckpt}"))
        #     print(Log.warning(f"No such pretrained model: {ep_model_path}, load from ckpt: {config.ckpt}"))
        else:
            model, vae, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=config.ckpt,
                                                                                      noise_sched_type=config.sched,
                                                                                      clip_sample=config.clip,
                                                                                      sde_type="SDE-LDM")
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        model, vae, noise_sched, get_pipeline = DiffuserModelSched.get_model_sched(image_size=dataset_loader.image_size,
                                                                              channels=dataset_loader.channel,
                                                                              model_type=DiffuserModelSched.LDM_CELEBA_HQ_DEFAULT,
                                                                              noise_sched_type=config.sched,
                                                                              clip_sample=config.clip)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    model = nn.DataParallel(model, device_ids=config.device_ids)

    lr_sched = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(dataset_loader.num_batch * config.epoch),
    )

    cur_epoch = cur_step = 0
    accelerator.register_for_checkpointing(model, vae, optimizer, lr_sched)

    return model, vae, optimizer, lr_sched, noise_sched, cur_epoch, cur_step, get_pipeline


def init_train(config: TrainingConfig, dataset_loader: DatasetLoader):
    # Initialize accelerator and tensorboard logging    
    accelerator = get_accelerator(config=config)
    repo = get_repo(config=config, accelerator=accelerator)

    model, vae, optimizer, lr_sched, noise_sched, cur_epoch, cur_step, get_pipeline = get_model_optim_sched(config=config,
                                                                                                       accelerator=accelerator,
                                                                                                       dataset_loader=dataset_loader)

    dataloader = dataset_loader.get_dataloader()
    model, vae, optimizer, dataloader, lr_sched = accelerator.prepare(
        model, vae, optimizer, dataloader, lr_sched
    )
    return accelerator, repo, model, vae, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline


def make_grid(images, rows, cols):
    w, h = images[0].size
    # print("image size", images[0].size)
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def make_grid_gray(images, rows, cols, idx=0):
    # Ensure the images are grayscale
    channels = images.shape[1]
    img_list = []
    for i in range(channels):
        img_list.append(Image.fromarray((images[idx][i] * 255).round().astype("uint8")))

    img_list = [img.convert('L') if img.mode != 'L' else img for img in img_list]

    w, h = img_list[0].size
    print("image size", img_list[0].size)
    grid = Image.new('L', size=(cols * w, rows * h))
    for i, image in enumerate(img_list):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def sampling(config: TrainingConfig, file_name: Union[int, str], pipeline):
    def gen_samples(init: torch.Tensor, folder: Union[os.PathLike, str]):
        test_dir = os.path.join(config.output_dir, folder)
        os.makedirs(test_dir, exist_ok=True)

        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        pipline_res = pipeline(
            batch_size=config.batch,
            generator=torch.manual_seed(config.seed),
            init=init,
            output_type=None,
            # save_every_step=True
        )
        print("seed: ", config.seed)
        images = pipline_res.images
        torch.save(images, f"sampling_{folder}.pt")
        movie = pipline_res.movie[0]
        activations = pipline_res.activations
        print('len', len(activations))  # 1000, 25
        print('movie shape', movie.shape)
        print('act shape', activations[0].shape)  # 16, 128, 32, 32
        # activations = [activation.sum(dim=1, keepdim=True) for activation in activations]

        # # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.

        if config.sched == "DDPM-SCHED":
            t_index = [999, 750, 500, 250, 0]
        elif config.sched == "DDIM-SCHED":
            t_index = [i for i in range(0, 25, 4)]
        print(t_index)

        images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
        image_grid = make_grid(images, rows=4, cols=4)

        r = int(pow(config.batch, 0.5))
        c = int(pow(config.batch, 0.5))
        r = 1
        c = 1
        print('r, c: ', r, c)
        clip_opt = "" if config.clip else "_noclip"
        for t in t_index:
            img = [Image.fromarray(image) for image in np.squeeze((movie[t] * 255).round().astype("uint8"))]
            grid_img = make_grid(img, rows=r, cols=c)

            # convout = [Image.fromarray(activation) for activation in
            #            np.squeeze((activations[t] * 255).round().astype("uint8"))]
            # grid_convout = make_grid(convout, rows=r, cols=c)
            if isinstance(file_name, int):
                print(1, f"{test_dir}/{file_name:04d}{clip_opt}_img_{25 - t}.png")
                image_grid.save(f"{test_dir}/{file_name:04d}{clip_opt}.png")
                grid_img.save(f"{test_dir}/{file_name:04d}{clip_opt}_img_{25 - t}.png")
                # grid_convout.save(f"{test_dir}/{file_name:04d}{clip_opt}_noise_{25 - t}.png")
                # sam_obj.save(file_path=f"{file_name:04d}{clip_opt}_samples.pkl")
                # sam_obj.plot_series(slice_idx=slice(None), end_point=True, prefix_img_name=f"{file_name:04d}{clip_opt}_sample_t", animate_name=f"{file_name:04d}{clip_opt}_movie", save_mode=Samples.SAVE_FIRST_LAST, show_mode=Samples.SHOW_NONE)
            elif isinstance(file_name, str):
                print(2, f"{test_dir}/{file_name}{clip_opt}.png")
                image_grid.save(f"{test_dir}/{file_name}{clip_opt}.png")
                grid_img.save(f"{test_dir}/{file_name}{clip_opt}_img_{25 - t}.png")
                grid_convout.save(f"{test_dir}/{file_name}{clip_opt}_noise_{25 - t}.png")
                # sam_obj.save(file_path=f"{file_name}{clip_opt}_samples.pkl")
                # sam_obj.plot_series(slice_idx=slice(None), end_point=True, prefix_img_name=f"{file_name}{clip_opt}_sample_t", animate_name=f"{file_name}{clip_opt}_movie", save_mode=Samples.SAVE_FIRST_LAST, show_mode=Samples.SHOW_NONE)
            else:
                raise TypeError(f"Argument 'file_name' should be string nor integer.")

        # sam_obj = Samples(samples=np.array(movie), save_dir=test_dir)
        print(folder)
        # activations_mean = [abs(activation.mean(axis=(2, 3))) for activation in activations] 
        activations_mean = [activation.mean(axis=(2, 3)) for activation in activations] 
        print(activations_mean[0].shape)

        flattened_data = [arr.flatten() for arr in activations_mean]
        with open(f'data_{folder}.txt', 'w') as f:
            for arr in flattened_data:
                line = ' '.join(map(str, arr))
                f.write(line + '\n')

    with torch.no_grad():
        noise = torch.randn(
            (config.batch, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size),
            generator=torch.manual_seed(config.seed),
        )
        # Sample Clean Samples
        gen_samples(init=noise, folder="samples")  # init is important, we must self-define it
        # Sample Backdoor Samples
        # init = noise + torch.where(dsl.trigger.unsqueeze(0) == -1.0, 0, 1)
        init = noise + dsl.trigger.unsqueeze(0)
        # np.save('init.npy', init)
        # print(f"Trigger - (max: {torch.max(dsl.trigger)}, min: {torch.min(dsl.trigger)}) | Noise - (max: {torch.max(noise)}, min: {torch.min(noise)}) | Init - (max: {torch.max(init)}, min: {torch.min(init)})")
        gen_samples(init=init, folder="backdoor_samples")


def save_imgs(imgs: np.ndarray, file_dir: Union[str, os.PathLike], file_name: Union[str, os.PathLike] = "") -> None:
    os.makedirs(file_dir, exist_ok=True)
    # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
    images = [Image.fromarray(image) for image in np.squeeze((imgs * 255).round().astype("uint8"))]
    for i, img in enumerate(tqdm(images)):
        img.save(os.path.join(file_dir, f"{file_name}{i}.png"))


"""With this in end, we can group all together and write our training function. This just wraps the training step we saw in the previous section in a loop, using Accelerate for easy TensorBoard logging, gradient accumulation, mixed precision training and multi-GPUs or TPU training."""


def get_ep_model_path(config: TrainingConfig, dir: Union[str, os.PathLike], epoch: int):
    return os.path.join(dir, config.ep_model_dir, f"ep{epoch}")


def checkpoint(config: TrainingConfig, accelerator: Accelerator, pipeline, cur_epoch: int, cur_step: int, repo=None,
               commit_msg: str = None):
    '''Only used in train_loop'''
    accelerator.save_state(config.ckpt_path)
    accelerator.save({'epoch': cur_epoch, 'step': cur_step}, config.data_ckpt_path)
    # if config.push_to_hub:
    #     push_to_hub(config, pipeline, repo, commit_message=commit_msg, blocking=True)
    # else:
    pipeline.save_pretrained(config.output_dir)

    if config.is_save_all_model_epochs:
        # ep_model_path = os.path.join(config.output_dir, config.ep_model_dir, f"ep{cur_epoch}")
        ep_model_path = get_ep_model_path(config=config, dir=config.output_dir, epoch=cur_epoch)
        os.makedirs(ep_model_path, exist_ok=True)
        pipeline.save_pretrained(ep_model_path)


def prepare_folder():
    """create folders for saving results."""
    current_file_path = __file__
    file_name = os.path.basename(current_file_path).split('.')[0]
    print(f"file name: {file_name}")  # BBTI_helper_villandiff
    model_name = format_ckpt_dir(config.ckpt)
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


def format_ckpt_dir(model_path):
    if '/' in model_path:
        return model_path.split('/')[-1]
    elif '\\' in model_path:
        return model_path.split('\\')[-1]
    return model_path


import torch.nn.functional as F
from PIL import Image
import itertools

EPSILON = 1e-07


# print(f"epsilon: {EPSILON}")

def get_raw_pattern(init_pattern):
    pattern = nn.Tanh()(init_pattern)
    return pattern  # [-1, 1]


def element_wize_grad(x, y):
    grads = torch.zeros_like(x)
    if x.grad is not None:
        x.grad.zero_()
    else:
        return grads
    shape_x = x.shape
    for i in range(shape_x[0]):
        for j in range(shape_x[1]):
            for k in range(shape_x[2]):
                y[i, j, k].backward(retain_graph=True)
                grads[i, j, k] = x.grad[i, j, k]
    return grads


def cosine_similarity(A, B):
    dot_product = torch.dot(A, B)
    norm_a = torch.norm(A)
    norm_b = torch.norm(B)
    return dot_product / (norm_a * norm_b)


def calculate_entropy(logits):
    # Apply softmax to convert logits into probabilities
    probabilities = F.softmax(logits, dim=-1)
    # Calculate the entropy
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9),
                         dim=-1)  # Adding a small value to avoid log(0)
    return entropy.mean()  # Return the mean entropy if you want a scalar value


def ste_round(x):  # round且保留梯度
    return torch.round(x) - x.detach() + x


def check_black_white_pixels(image_tensor):
    gray_image = image_tensor.mean(dim=0)

    black_pixels = (gray_image <= 0.01).sum()
    white_pixels = (gray_image >= 0.99).sum()
    total_pixels = gray_image.numel()
    black_persentage = black_pixels / total_pixels
    white_percentage = white_pixels / total_pixels
    if black_persentage > 0.5:
        return -1
    elif white_percentage > 0.5:
        return 1
    else:
        return 0


def calc_loss_entropy(target_set, residual_set, images):
    loss_x0_entropy_weight = 10

    loss_x0_entropy = 0
    if target_set == [] or calculate_entropy(
            ste_round(images[target_set[0]] * 255)) < 0.8: 
        entropy_threshold_max = 6
        entropy_threshold_min = 0.8
        loss_x0_entropy_weight *= 2.
    else:
        entropy_list = []
        for i in target_set:
            entropy_list.append(calculate_entropy(ste_round(images[i] * 255)))
        entropy_threshold_max = max(entropy_list) + 0.1
        entropy_threshold_min = min(entropy_list) - 0.1

    for i in residual_set:
        entropy_loss_tmp = calculate_entropy(ste_round(images[i] * 255))  
        # print(f"i: {entropy_loss_tmp}")
        loss_x0_entropy += torch.relu(entropy_loss_tmp - entropy_threshold_max) + torch.relu(
            entropy_threshold_min - entropy_loss_tmp)
    return loss_x0_entropy_weight, loss_x0_entropy


def calc_loss_similarity(images):
    combinations = list(itertools.combinations(range(config.batch), 2)) 
    target_set = [] 
    similarities_all = []
    for (a, b) in combinations:
        if check_black_white_pixels(images[a]) != 0 or check_black_white_pixels(images[b]) != 0:
            continue
        similarity = cosine_similarity(images[a], images[b])
        similarities_all.append(similarity)
        if similarity > 0.999:
            if not a in target_set:
                target_set.append(a)
            if not b in target_set:
                target_set.append(b)

    residual_set = []
    for i in range(config.batch):
        if not i in target_set:
            residual_set.append(i)

    # calculate the similarity loss
    similarities = []
    if target_set != []:
        target_img = images[target_set[0]]
        for i in residual_set:
            similarity = cosine_similarity(target_img, images[i])  
            similarities.append(similarity)
    else:
        similarities = similarities_all
    if similarities == []:
        loss_s = 0
    else:
        loss_s = len(similarities) - torch.sum(torch.stack(similarities))  # f(x0)
    return target_set, residual_set, loss_s


def trigger_inversion(config: TrainingConfig, accelerator: Accelerator, repo, model: nn.Module, vae, get_pipeline,
                      noise_sched,
                      optimizer: torch.optim, loader, lr_sched, start_epoch: int = 0, start_step: int = 0):
    input_path, trigger_path, img_path, score_json_path = prepare_folder()
    model_name = format_ckpt_dir(config.ckpt)

    torch.manual_seed(0)
    device = f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu"
    # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
    pipeline = get_pipeline(accelerate=accelerator, unet=accelerator.unwrap_model(model), vae=vae,
                            scheduler=noise_sched)
    # print("pipeline: ", pipeline)

    global dsl
    # dataloader = dsl.get_dataloader()
    # for batch in dataloader:
    #     clean_images = batch['pixel_values']
    #     print(clean_images.shape)
    #     break
    target = dsl.target
    trigger = dsl.trigger
    print(f"target mean: {torch.mean(target)}, target var: {torch.var(target)}")
    print(target.shape)

    if "CIFAR10" in config.ckpt:  # hyperparameters
        
        init_pattern = torch.randn((3, 32, 32), dtype=torch.float, device=device, requires_grad=True)

    
        print(init_pattern.shape)
        
        init_pattern = init_pattern.to(device)
        init_pattern.requires_grad_(True)
        # init_pattern = torch.zeros((3, 32, 32), dtype=torch.float, device=device, requires_grad=True)
        grad_scalar = 1e3  # shift-hat用的1e3
        lr = 1e-1  # stopsign14-shift用lr=1e-1
        loss_g_entropy_weight = 0
    elif "CELEBA-HQ" in config.ckpt:
        # init_pattern = torch.randn((3, 256, 256), dtype=torch.float, device=device, requires_grad=True)
        # init_pattern = copy.deepcopy(target)
        init_pattern = torch.randn((3, 64, 64), dtype=torch.float, device=device, requires_grad=True)
        # init_pattern = init_pattern.to(device)
        # init_pattern.requires_grad_(True)
        lr = 1e-1
        grad_scalar = 10
        loss_g_entropy_weight = 0.1

    # init_pattern.requires_grad_(True)
    noise = torch.randn((config.batch, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size),
                        generator=torch.manual_seed(config.seed))
    print(pipeline)
    noise = noise.to(device)
    optimizer = torch.optim.Adam([init_pattern], lr=lr, betas=(0.9, 0.999))  # nc的代码里用的值是0.5,0.9


    grad_x0_xt = torch.tensor(1.)
    grad_x0_xt = grad_x0_xt.to(device)
    # print(f"grad x0xt: {grad_x0_xt}")  # 0.0064
    # grad_xt_trigger = sqrt_one_minus_alpha_prod.to(device)
    # grad_xt_trigger = 1 - sqrt_alpha_prod.to(device)

    row_number = int(math.sqrt(config.batch))
    best_pattern = init_pattern.clone  
    best_similar_num = 0  
    best_span = 0  

    print(f"min pattern: {torch.min(init_pattern)}, max pattern: {torch.max(init_pattern)}")
    flag = 0
    # init_pattern = init_pattern - 0.05
    unet = pipeline.unet.cuda()
    scheduler = pipeline.scheduler
    scheduler.set_timesteps(20)
    timesteps = scheduler.timesteps

    for epoch in range(100):  # the reverse process
        optimizer.zero_grad()
        # pattern = get_raw_pattern(init_pattern)  

        pattern = -1. * init_pattern
        pattern = pattern.to(device)
        pattern = ste_round(pattern * 255.) / 255.  
        x_T = noise + pattern

        # zero_noise = torch.zeros_like(noise)
        # x_T = pattern + zero_noise
        # x_T = noise
        x_T.requires_grad_(True)

        # if epoch == 1:
        #     x_T = copy.deepcopy(images)
        #     x_T.requires_grad_(True)

        pipline_res = pipeline(batch_size=config.batch, num_inference_steps = 20,
            generator=torch.manual_seed(config.seed), init=x_T, output_type=None)
        images = pipline_res.images  
        images = images.transpose(0, 3, 1, 2)  
        images = torch.from_numpy(images)
        images = images.to(device)
        images.requires_grad_(True)
        images_flattened = images.reshape(config.batch, -1)

        # calculate entropy loss of pattern
        g_entropy = calculate_entropy(ste_round(pattern * 255))
        entropy_loss_g = loss_g_entropy_weight * (
                    torch.relu(g_entropy - 3.) + torch.relu(1. - g_entropy))  
        # calculate similarity loss
        target_set, residual_set, loss_s = calc_loss_similarity(images_flattened)
        if residual_set != []:
            loss_s_weight = config.batch / len(residual_set) 
        else:
            loss_s_weight = 1

        loss_x0_entropy_weight, loss_x0_e = calc_loss_entropy(target_set, residual_set, images_flattened)

        value_loss = 0
        for i in residual_set:
            value_loss += torch.relu(images[i] - 0.95).sum() + torch.relu(0.05 - images[i]).sum()
        # if value_loss > 100:
        #     value_loss /= 100
        # elif value_loss > 10:
        #     value_loss /= 10
        print(type(loss_s), loss_s, type(value_loss), value_loss)
        if loss_s != 0 and value_loss != 0:
            # loss_x0 = - loss_s_weight * loss_s + value_loss
            loss_x0 = loss_s_weight * loss_s + loss_x0_entropy_weight * loss_x0_e + value_loss
            # loss_x0 = loss_s_weight * loss_s + value_loss_weight * value_loss
            loss_x0.backward()
            x_T.grad = grad_x0_xt * images.grad  # 16, 3, 32, 32
            pattern.grad = torch.mean(x_T.grad, dim=0, keepdim=False)  # 对batchsize维度求均值，这个操作是没问题的.用一个batch更新参数时，确实应该取平均
            grad_initp_p = grad_scalar * element_wize_grad(init_pattern, pattern)
            # print(f"grad initp p shape: {grad_initp_p.shape}, pattern grad shape:{pattern.grad.shape}")
            init_pattern.grad = grad_initp_p * pattern.grad
            entropy_loss_g.backward(retain_graph=True)  # entropy loss一步到位
            optimizer.step()

        if len(target_set) > best_similar_num:
            best_similar_num = len(target_set)
            best_pattern = init_pattern.detach().clone()
            best_span = 0


        asr = len(target_set) / config.batch
        if epoch % 1 == 0:
            # calculate SSIM score
            images = images.to(device)
            reps = ([images.shape[0]] + ([1] * (len(dsl.target.shape))))
            backdoor_target = torch.squeeze((dsl.target.repeat(*reps) / 2 + 0.5).clamp(0, 1)).to(device)
            ssim_sc = float(
                StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(images, backdoor_target))
            print(f"Epoch {epoch} "
                  f"ssim: {ssim_sc}, asr: {asr}")
            if "ASR" not in score_dict[model_name].keys() or asr > score_dict[model_name]["ASR"]:
                score_dict[model_name]["ASR"] = asr
                score_dict[model_name]["SSIM"] = ssim_sc
                with open(score_json_path, 'w') as file:
                    json.dump(score_dict, file, indent=4)



        if epoch % config.save_epoch == 0 or loss_s == 0:
            # save x_0
            torch.save(images.detach().cpu(), osp.join(img_path, f"epoch{epoch:06d}_x_0.pt"))
            print(f"Epoch {epoch} - images mean: {torch.mean(images)} - images var: {torch.var(images)}")
            print(f"Epoch {epoch} - pattern mean: {torch.mean(init_pattern)} - pattern var: {torch.var(init_pattern)}")
            print(f"number similar: {len(target_set)}, best answer: {best_similar_num}")
            # save x0 in grid
            x0_images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
            x0_images = [Image.fromarray(image) for image in np.squeeze((x0_images * 255).round().astype("uint8"))]
            x0_image_grid = make_grid(x0_images, rows=row_number, cols=row_number)
            x0_image_grid.save(osp.join(img_path, f"img_epoch{epoch:06d}_x_0.png"))
            # print(f"grad_p: {grad_p.cpu().numpy()}")

            # save trigger pattern
            np.save(osp.join(trigger_path, f"trigger_epoch{epoch:06d}.npy"), pattern.detach().cpu().numpy())
            pattern_np = pattern.permute(1, 2, 0).detach().cpu().numpy()
            # print(f"Epoch {epoch} - step {i} - pattern mean: {np.mean(pattern_np)} - pattern var: {np.var(pattern_np)}")
            pattern_np = (pattern_np * 255).round().astype("uint8")
            print(f"trigger shape: {pattern_np.shape}")
            pattern_image = Image.fromarray(pattern_np)
            pattern_image.save(osp.join(trigger_path, f"trigger_epoch{epoch:06d}.png"))

            noise_np = noise.permute(0, 2, 3, 1).detach().cpu().numpy()
            # print("noise ", noise_plus_trigger.shape)
            xT_images = [Image.fromarray(noise) for noise in
                         np.squeeze((noise_np * 255).round().astype("uint8"))]
            image_grid = make_grid(xT_images, rows=row_number, cols=row_number)
            image_grid.save(osp.join(input_path, f"input_epoch{epoch:06d}.png"))

        if asr == 1. or loss_s == 0:
            print(asr, loss_s)
            break
            #################################
            # save movie
            # movies = pipline_res.movie  # movie是一个长度为T的list，每个元素都是bz,3,32,32的图
            # print(f"movies shape: {movies.shape}")
            # movie_images = []
            # row_number = 2
            # for t in range(len(movies)):
            #     movie_img = np.squeeze((movies[t][0] * 255).round().astype("uint8"))
            #     movie_images.append(Image.fromarray(movie_img))
            #
            # movie_grid = make_grid(movie_images[::-1], rows=row_number, cols=len(movies) // row_number)
            # movie_grid.save(osp.join(img_path, f"movie_epoch{epoch:06d}.png"))



    return pipeline


if __name__ == '__main__':
    config = setup()
    dsl = get_data_loader(config=config)
    # print("trigger shape: ", dsl.trigger.shape)  # 3, 32, 32
    accelerator, repo, model, vae, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_train(
        config=config, dataset_loader=dsl)

    if config.mode == MODE_REVERSE:
        score_dict = {}
        pipeline = trigger_inversion(config, accelerator, repo, model, vae, get_pipeline, noise_sched, optimizer, dataloader,
                                     lr_sched,
                                     start_epoch=cur_epoch, start_step=cur_step)
    else:
        raise NotImplementedError()

    accelerator.end_training()

# python BBTI_helper_ldm.py --project 1 --mode reverse --ckpt D:\BackdoorDiff-DS\VillanDiff\ldm\res_LDM-CELEBA-HQ-256_CELEBA-HQ-LATENT_ep2000_ode_c1.0_p0.9_GLASSES-CAT_psi1.0_new-set --sched UNIPC-SCHED --fclip o --gpu 0 --batch 16 --save_epoch 1 -is 20

