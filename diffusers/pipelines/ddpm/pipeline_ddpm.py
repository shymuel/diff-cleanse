# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# limitations under the License.

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np

import pandas as pd
import re

from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
import copy


class DDPMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 16,
            generator: Optional[torch.Generator] = None,
            num_inference_steps: int = 1000,
            output_type: Optional[str] = "numpy",
            return_dict: bool = True,
            init: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """

        if init == None:
            # Sample gaussian noise to begin loop
            # image = torch.randn(
            #     (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            #     generator=generator,
            # )
            image = randn_tensor((batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator, device=self.device, dtype=self.unet.dtype)
        else:
            image = init.detach().clone()
        image = image.to(self.device)

        # set step values
        print(self.scheduler)
        self.scheduler.set_timesteps(num_inference_steps)

        mov = [None] * len(self.scheduler.timesteps) 
        for t in self.progress_bar(self.scheduler.timesteps):  # t从999到0
            # 1. predict noise model_output
            model_output = self.unet(image, t, return_dict=True).sample
            # 2. compute previous image: x_t -> t_t-1
            # image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample
            image = self.scheduler.step(model_output, t, image).prev_sample
            # mov[t] = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            mov = list(map(self.numpy_to_pil, mov))
        if not return_dict:
            return image, mov, []
        # movie is the mid progress, this is modified by the paper writer
        return ImagePipelineOutput(images=image, movie=mov, activations=[])



class DDPMPipeline_PRUNE(DiffusionPipeline): 
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 1,
            generator: Optional[torch.Generator] = None,
            num_inference_steps: int = 1000,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            init: Optional[torch.Tensor] = None,
            prune_idx_list: list = [],
            prune_threshold: int = 0,
            target_up_block: int = 3,
            **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """

        # def hook_fn(module, input, output):
        #     output_t = torch.mean(output, dim=0)
        #     act_hook.append(output_t.cpu().detach().numpy() / 10)


        if init == None:
            # Sample gaussian noise to begin loop
            image = randn_tensor((batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator, device=self.device, dtype=self.unet.dtype)
            # image = torch.randn(
            #     (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            #     generator=generator,
            # )
        else:
            image = init.detach().clone()
        image = image.to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        label = "last_block"
        # label = "conv_act"

        mov = [None] * len(self.scheduler.timesteps)
        targetLayerActivationAll = [None] * len(self.scheduler.timesteps)
        if prune_idx_list != []:
            split_span = 1000 // len(prune_idx_list)
            print(f"ddpm prune split_span: {split_span}")
        else:
            print("prune index list is empty!!!")

        params_init = copy.deepcopy(self.unet.state_dict())

        for t in self.progress_bar(self.scheduler.timesteps):
            if prune_idx_list != []:
                if (t + 1) % split_span == 0 and (t+1) >= prune_threshold:  # split span = 1000 or 1
                # if (t + 1) % split_span == 0 and (t + 1) < prune_threshold:
                    self.unet.load_state_dict(params_init) 
                    # print(f"t: {t}, prune index: {prune_idx_list[((t + 1) // split_span) - 1]}")
                    for idx in prune_idx_list[((t + 1) // split_span) - 1]:
                        if label == "last_block":
                            # self.unet.up_blocks[3].resnets[2].conv_shortcut.weight[:, idx, :, :] = 0
                            self.unet.up_blocks[target_up_block].resnets[2].conv_shortcut.weight[idx, :, :, :] = 0
                            self.unet.up_blocks[target_up_block].resnets[2].conv_shortcut.bias[idx] = 0
                            self.unet.up_blocks[target_up_block].resnets[2].conv2.weight[idx, :, :, :] = 0
                            self.unet.up_blocks[target_up_block].resnets[2].conv2.bias[idx] = 0
                        elif label == "conv_act":
                            self.unet.conv_norm_out.weight[idx] = 0
                            self.unet.conv_norm_out.bias[idx] = 0
                if (t+1) == prune_threshold - 1:
                    self.unet.load_state_dict(params_init)  

            # 1. predict noise model_output
            model_output = self.unet(image, t).sample
            # 2. compute previous image: x_t -> t_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample


        # hook.remove()
        self.unet.load_state_dict(params_init)

        image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            mov = list(map(self.numpy_to_pil, mov))
            targetLayerActivationAll = list(map(self.numpy_to_pil, targetLayerActivationAll))
        if not return_dict:
            return image, mov, targetLayerActivationAll

        return ImagePipelineOutput(images=image, movie=mov,
                                   activations=targetLayerActivationAll)  # movie is the mid progress, this is modified by the paper writer
