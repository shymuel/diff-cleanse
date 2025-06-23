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

from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--output_path", type=str, default="exp", help="Path for saving running related data.")
    parser.add_argument("--doc", type=str, default='ddpm', help="model name.")
    parser.add_argument("--use_ema", action="store_true", default=False, help="Use EMA.")
    parser.add_argument("--comment", type=str, default="", help="A string for experiment comment")
    parser.add_argument("--verbose", type=str, default="info",
        help="Verbose level: info | debug | warning | critical")
    parser.add_argument("--measure", action="store_true", help="Whether to measure the model")
    parser.add_argument("--sample", action="store_true",
        help="Whether to produce samples from the model")
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--resume_training", action="store_true", help="Whether to resume training")
    parser.add_argument("-i", "--image_folder", type=str, default="images",
        help="The folder name of samples")
    parser.add_argument("--ni", action="store_true",
        help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--sample_type", type=str, default="generalized",
        help="sampling approach (generalized or ddpm_noisy)")
    parser.add_argument("--skip_type", type=str, default="uniform",
        help="skip according to (uniform or quadratic)")
    parser.add_argument("--timesteps", type=int, default=1000, help="number of steps involved")
    parser.add_argument("--eta", type=float, default=0.0,
        help="eta used to control the variances of sigma")
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--loss_p", action="store_true", help="Use the proposed loss_p, use unpruned model as teacher model")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # 加载模型
    parser.add_argument("--model_path", type=str, help="the file path of the unpruned model")
    parser.add_argument("--pruned_model", type=str, help="the file path of the pruned model")

    # accelerator
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"],
                        help=(
                            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
                            " for experiment tracking and logging of model metrics and model checkpoints"))
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help=("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
                              " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."))
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
                        help=("Whether to use mixed precision. Choose"
                              "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
                              "and an Nvidia Ampere GPU."))
    parser.add_argument("--prediction_type", type=str,
                        default="epsilon", choices=["epsilon", "sample"],
                        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.")

    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.doc)  # attack
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    args.log_path = args.output_path

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    if not args.measure and not args.sample:
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
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            os.makedirs(os.path.join(args.output_path, "image_samples"), exist_ok=True)
            args.image_folder = os.path.join(args.output_path, "image_samples", args.image_folder)
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                if not (args.fid or args.interpolation):
                    overwrite = False
                    if args.ni:
                        overwrite = True
                    else:
                        response = input(
                            f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                        )
                        if response.upper() == "Y":
                            overwrite = True

                    if overwrite:
                        shutil.rmtree(args.image_folder)
                        os.makedirs(args.image_folder)
                    else:
                        print("Output image folder exists. Program halted.")
                        sys.exit(0)
        elif args.measure:
            os.makedirs(os.path.join(args.exp, "measure"), exist_ok=True)
            args.image_folder = os.path.join(args.exp, "measure")
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    try:
        runner = Diffusion(args, config)
        if args.sample:
            runner.sample()
        elif args.measure:
            runner.measure(num_clean=10000, num_backdoor=10)
        else:
            runner.train(loss_p=args.loss_p)
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
