# VillanDiffusion
"VillanDiffusion: A Unified Backdoor Attack Framework for Diffusion Models" published at NeurIPS 2023. The official repo is [VillanDiff](https://github.com/IBM/VillanDiffusion).

VillanDiff can attack DDPM, Score-baded DM (NCSN in the official code) and LDM. For DDPM and LDM models, VillanDiff supports advanded ODE samplers, but only support SDE sampler for NCSN models. The original paper tests VillanDiff on CIFAR10 and CelebA-HQ 256 datasets.
We provice codes for training, sampling and measuring. 


## Training
```

```

## Sampling
```
python baddiffusion.py --project default --mode sampling --ckpt [absolute path of the model] --sched DDPM-SCHED --fclip o --gpu 0
```

## Measure
```
python baddiffusion.py --project default --mode measure --ckpt [absolute path of the model] -sc DDPM-SCHED --fclip o -o --gpu 0
```