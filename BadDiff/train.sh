# finetune-cifar10
python baddiffusion.py --project default --mode train --dataset CIFAR10 --batch 128 --epoch 50 --poison_rate 0.1 --trigger BOX_14 --target HAT --ckpt DDPM-CIFAR10-32 -sc DDPM-SCHED --fclip o -o --gpu 0

# finetune-cifar10
python baddiffusion.py --project default --mode train --dataset CIFAR10 --batch 128 --epoch 50 --poison_rate 0.1 --trigger BOX_14 --target HAT --ckpt DDPM-CIFAR10-32 -sc DDPM-SCHED --fclip o -o --gpu 0,1,2,3


# training-from-scratch
python baddiffusion.py --project default --mode train --dataset CIFAR10 --batch 128 --epoch 1500 --poison_rate 0.0 --trigger BOX_14 --target TRIGGER -sc DDPM-SCHED --fclip o -o --gpu 0

# multi-gpu
accelerate launch --mixed_precision fp16 --num_processes=4 baddiffusion.py --project default --mode train --dataset CIFAR10 --batch 64 --epoch 50 --poison_rate 0.1 --trigger BOX_14 --target CAT -sc DDPM-SCHED --fclip o -o --gpu 0,1,2,3

# finetune-celebahq
python baddiffusion.py --project default --mode train --dataset CELEBA-HQ --batch 10 --epoch 50 --poison_rate 0.5 --trigger GLASSES --target CAT --ckpt DDPM-CELEBA-HQ-256 -sc DDPM-SCHED --fclip o -o --gpu 0