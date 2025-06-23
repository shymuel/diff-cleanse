# prune for trojdiff

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
from tqdm import tqdm
from runners_attack.diffusion import Diffusion
from torchvision import transforms
import torchvision
from custom_datasets import get_dataset, data_transform, inverse_data_transform
import torchvision.utils as tvu
from utils import UnlabeledImageFolder
import copy

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=2333, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for taylor expansion")
    parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data.")
    parser.add_argument("--doc", type=str, required=True,
                        help="A string for documentation purpose. Will be the name of the log folder.")
    parser.add_argument("--comment", type=str, default="", help="A string for experiment comment")

    parser.add_argument("--pruned_model", type=str, default=None, help="load pruned models, to sample or measure")
    parser.add_argument("--save_path", type=str, default=None, help="save path of results")

    parser.add_argument("--verbose", type=str, default="info", help="Verbose level: info | debug | warning | critical")
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument("--sample", action="store_true", help="Whether to produce samples from the model")
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--resume_training", action="store_true", help="Whether to resume training")
    parser.add_argument("-i", "--image_folder", type=str, default="images", help="The folder name of samples")
    parser.add_argument("--ni", action="store_true", help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument("--use_generated_samples", action="store_true",
                        help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument("--use_ema", action="store_true", help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--sample_type", type=str, default="generalized",
                        help="sampling approach (generalized or ddpm_noisy)")
    parser.add_argument("--skip_type", type=str, default="uniform", help="skip according to (uniform or quadratic)")

    parser.add_argument("--pruner", type=str, default="taylor",
                        choices=["taylor", "random", "magnitude", "reinit", "first_order_taylor", "second_order_taylor",
                                 'abs_taylor', 'fisher', 'diff-pruning', 'diff-cleanse'])

    parser.add_argument("--restore_from", type=str, default=None, help="Restore from user a checkpoint")
    parser.add_argument("--timesteps", type=int, default=1000, help="number of steps involved")
    parser.add_argument("--eta", type=float, default=0.0, help="eta used to control the variances of sigma")
    parser.add_argument("--thr", type=float, default=0.01, help="eta used to control the variances of sigma")
    parser.add_argument("--pruning_ratio_start", type=float, default=0.)
    parser.add_argument("--pruning_ratio", type=float, default=0.0, help="pruning ratio")
    parser.add_argument("--goal", type=int, default=-1,
                        help="1: remove the least important; -1: remove the most important")
    parser.add_argument("--threshold", type=int, default=950,
                        help="the start point of gradients accumulation.")

    parser.add_argument("--sequence", action="store_true")

    # attack
    parser.add_argument('--cond_prob', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--target_label', type=int, default=7)
    parser.add_argument('--miu_path', type=str, default='images/hello_kitty.png')
    parser.add_argument('--trigger_type', type=str, default='blend')
    parser.add_argument('--patch_size', type=int, default=3)
    parser.add_argument('--set', type=str, default='')

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

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
    def sample(model, save_path=""):
        with torch.no_grad():
            n = config.sampling.batch_size
            noise = torch.randn(n, config.data.channels, config.data.image_size,
                                config.data.image_size, device=runner.device)
            x = runner.sample_image(noise, model)
            x = inverse_data_transform(config, x)
            tvu.save_image(tvu.make_grid(x), save_path)

    def sample_bd(model, save_path=""):
        with torch.no_grad():
            n = config.sampling.batch_size
            x = torch.randn(n, config.data.channels, config.data.image_size,
                            config.data.image_size, device=runner.device)
            miu = torch.stack([runner.miu.to(noise.device)] * n)  # (batch,3,32,32)
            tmp_x = x.clone()  # 原图
            x = args.gamma * x + miu  # N(miu,I)
            if args.trigger_type == 'patch':
                tmp_x[:, :, -args.patch_size:, -args.patch_size:] = \
                    x[:, :, -args.patch_size:, -args.patch_size:]  # 只有patch对应的部分修改了
                x = tmp_x  # 赋值给x

            x = runner.sample_image_bd(x, model)
            x = inverse_data_transform(config, x)
            tvu.save_image(tvu.make_grid(x), save_path)

    args, config = parse_args_and_config()
    args.save_path = os.path.join(args.save_path, args.restore_from.split('/')[-4], args.doc, f"{args.pruner}")
    runner = Diffusion(args, config)
    if args.pruning_ratio > 0 and args.pruned_model is None:
        # Dataset 
        print(config)
        dataset, _ = get_dataset(args, config)
        print(f"Dataset size: {len(dataset)}")
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
        )
        from models.diffusion import AttnBlock
        import torch_pruning as tp
        print("Pruning ...")
        model = runner.model.eval()
        model.to(runner.device)
        example_inputs = {'x': torch.randn(1, 3, config.data.image_size, config.data.image_size).to(runner.device),
                          't': torch.ones(1).to(runner.device)}
        if args.pruner == 'taylor':
            imp = tp.importance.TaylorImportance()
        # elif args.pruner == 'first_order_taylor':
        #     imp = tp.importance.FullTaylorImportance(order=1)
        # elif args.pruner == 'second_order_taylor':
        #     imp = tp.importance.FullTaylorImportance(order=2)
        elif args.pruner == 'random' or args.pruner == 'reinit':
            imp = tp.importance.RandomImportance()
        elif args.pruner == 'magnitude':
            imp = tp.importance.MagnitudeImportance()
        # elif args.pruner == 'abs_taylor':
        #     imp = tp.importance.AbsTaylorImportance()
        # elif args.pruner == 'fisher':
        #     imp = tp.importance.FisherImportance()
        elif args.pruner == 'diff-pruning':
            imp = tp.importance.TaylorImportance(multivariable=False, goal=args.goal)
        elif args.pruner == 'diff-cleanse':
            imp = tp.importance.TaylorImportance(
                multivariable=False,
                goal=args.goal)  # a modified version, estimating the accumulated error of weight removal

        ignored_layers = [model.conv_out]
        channel_groups = {}
        iterative_steps = 1
        pruner = tp.pruner.MagnitudePruner(model, example_inputs,
            importance=imp, iterative_steps=iterative_steps,
            channel_groups=channel_groups, pruning_ratio_start=args.pruning_ratio_start, pruning_ratio=args.pruning_ratio,
            # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers, root_module_types=[torch.nn.Conv2d, torch.nn.Linear])

        # torch.manual_seed(10)
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)

        n = config.sampling.batch_size
        noise = torch.randn(n, config.data.channels, config.data.image_size,
                            config.data.image_size, device=runner.device)

        # sample before pruning
        sample_folder_path = os.path.join(args.save_path,
                                          f"vis{f'_{args.pruning_ratio_start}' if args.pruning_ratio_start > 0 else ''}"
                                          f"_{args.pruning_ratio}")
        os.makedirs(sample_folder_path, exist_ok=True)  # create the 'vis' folder
        sample(model, save_path=os.path.join(sample_folder_path, 'before_pruning_clean.png'))
        sample_bd(model, save_path=os.path.join(sample_folder_path, 'before_pruning_backdoor.png'))

        init_model = copy.deepcopy(model)
        if 'taylor' in args.pruner or 'fisher' in args.pruner or 'diff-pruning' in args.pruner or 'diff-cleanse' in args.pruner:
            x = iter(train_dataloader).next()
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(runner.device)
            x = data_transform(config, x)
            x = x.to(runner.device)
            n = x.size(0)
            e = torch.randn_like(x)
            b = runner.betas
            from functions.losses import noise_estimation_loss, noise_estimation_loss_bd
            model.zero_grad()
            max_loss = 0
            print("Accumulating gradients for pruning clean inputs")  # 积累梯度
            threshold = args.threshold
            for step_k in tqdm(range(threshold, 1000)):
                t = torch.ones(n, dtype=torch.long).to(runner.device) * step_k
                loss = noise_estimation_loss(model, x, t, e, b)
                if args.pruner == 'diff-pruning':
                    if loss > max_loss:
                        max_loss = loss
                    if loss < max_loss * args.thr:
                        break
                    # print(loss, max_loss)
                loss.backward()

            # print(model)
            for g in pruner.step(interactive=True):
                g.prune()


            if args.pruner == 'diff-cleanse':
                model = copy.deepcopy(init_model)
                model.zero_grad()  # 模型梯度清0
                model.eval()
                print("type importance: ", type(pruner.importance_results))
                if type(pruner.importance_results) == type([]):
                    print("len: ", len(pruner.importance_results))  # 50
                for l in range(len(pruner.importance_results)):
                    print(pruner.importance_results[l].shape, end=", ")
                print("")
                init_importance = pruner.importance_results
                ignored_layers = [model.conv_out]
                channel_groups = {}
                pruner = tp.pruner.MagnitudePruner(model, example_inputs,
                    importance=imp, iterative_steps=1,
                    channel_groups=channel_groups, pruning_ratio_start=args.pruning_ratio_start, pruning_ratio=args.pruning_ratio,
                    ignored_layers=ignored_layers, init_importance=init_importance)

                # for g in pruner.DG.get_all_groups(ignored_layers=ignored_layers, root_module_types=pruner.root_module_types):
                #     print(g)

                max_loss = 0
                print("Accumulating gradients for pruning backdoored inputs")  # 积累梯度
                for step_k in tqdm(range(threshold, 1000)):
                    t = torch.ones(n, dtype=torch.long).to(runner.device) * step_k
                    loss = noise_estimation_loss_bd(model, x, t, e, b, runner.miu, args)
                    if args.pruner == 'diff-pruning':
                        if loss > max_loss:
                            max_loss = loss
                        if loss < max_loss * args.thr:
                            break
                        # print(loss, max_loss)
                    loss.backward()


        for g in pruner.step(interactive=True):
            # print("type g: ", type(g))  # torch_pruning.dependency.Group
            g.prune()

        if args.pruner == 'reinit':
            def reset_parameters(model):
                for m in model.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()

            model.apply(reset_parameters)

        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print("============ After Pruning ============")
        # print(model)
        print("#Params: {:.4f} M => {:.4f} M".format(base_nparams / 1e6, nparams / 1e6))
        print("#MACs: {:.4f} G => {:.4f} G".format(base_macs / 1e9, macs / 1e9))
        del pruner
        # Save pruned model
        os.makedirs(args.save_path, exist_ok=True)
        torch.save(model, os.path.join(args.save_path, f"unet_pruned{f'_{args.pruning_ratio_start}_{threshold}' if args.pruning_ratio_start>0 else ''}"
                                                       f"_{args.pruning_ratio}_{threshold}.pth"))

        # sample
        sample(model, save_path=os.path.join(sample_folder_path, f'after_pruning_clean{threshold}.png'))
        sample_bd(model, save_path=os.path.join(sample_folder_path, f'after_pruning_backdoor{threshold}.png'))

    return 0


if __name__ == "__main__":
    sys.exit(main())
