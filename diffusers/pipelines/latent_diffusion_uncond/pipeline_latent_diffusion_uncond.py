# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import inspect
from typing import List, Optional, Tuple, Union

import torch

from ...models import UNet2DModel, VQModel
from ...schedulers import DDIMScheduler
from ...utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class LDMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        vqvae ([`VQModel`]):
            Vector-quantized (VQ) Model to encode and decode images to and from latent representations.
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            [`DDIMScheduler`] is to be used in combination with `unet` to denoise the encoded image latents.
    """

    def __init__(self, vqvae: VQModel, unet: UNet2DModel, scheduler: DDIMScheduler, clip_sample=False):
        super().__init__()
        self.register_modules(vqvae=vqvae, unet=unet, scheduler=scheduler)
        self.clip_sample = clip_sample

    @staticmethod
    def __encode_latents(vae: VQModel, x: torch.Tensor, weight_dtype: str = None, scaling_factor: float = None):
        x = x.to(vae.device)
        if weight_dtype != None and weight_dtype != "":
            x = x.to(dtype=weight_dtype)
        if scaling_factor != None:
            return vae.encode(x).latents * scaling_factor
        # return vae.encode(x).latents * vae.config.scaling_factor
        return vae.encode(x).latents

    @staticmethod
    def __decode_latents(vae: VQModel, x: torch.Tensor, weight_dtype: str = None, scaling_factor: float = None):
        x = x.to(vae.device)
        if weight_dtype != None and weight_dtype != "":
            x = x.to(dtype=weight_dtype)
        if scaling_factor != None:
            return vae.decode(x).sample / scaling_factor
        # return vae.decode(x).sample / vae.config.scaling_factor
        return vae.decode(x).sample

    def encode(self, image: torch.Tensor, weight_dtype: str = None, scaling_factor: float = None, *args, **kwargs):
        return LDMPipeline.__encode_latents(vae=self.vqvae, x=image, weight_dtype=weight_dtype,
                                            scaling_factor=scaling_factor)

    def decode(self, image: torch.Tensor, weight_dtype: str = None, scaling_factor: float = None, *args, **kwargs):
        return LDMPipeline.__decode_latents(vae=self.vqvae, x=image, weight_dtype=weight_dtype,
                                            scaling_factor=scaling_factor)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        init: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        save_every_step: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """

        if init == None:
            latents = randn_tensor(
                (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
                generator=generator,
            )
            latents = latents.to(self.device)
        else:
            latents = init.detach().clone().to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        mov = [None] * len(self.scheduler.timesteps)
        timestep_idx = len(self.scheduler.timesteps) - 1
        if save_every_step:
            mov[-1] = (self.vqvae.decode(latents).sample / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        for t in self.progress_bar(self.scheduler.timesteps):
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            # predict the noise residual
            noise_prediction = self.unet(latent_model_input, t).sample
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_prediction, t, latents, **extra_kwargs).prev_sample
            # print("latents: ", latents.shape)
            if self.clip_sample:
                latents = latents.clamp(-1, 1)
            if save_every_step:
                pass
                # timestep_idx -= 1
                # mov[timestep_idx] = (self.vqvae.decode(latents).sample / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

        # decode the image latents with the VAE
        image = self.vqvae.decode(latents).sample
        # print("image shape: ", image.shape)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            if save_every_step:
                mov = list(map(self.numpy_to_pil, mov))

        if not return_dict:
            return (image, mov, [])

        return ImagePipelineOutput(images=image, movie=mov, activations=[])

