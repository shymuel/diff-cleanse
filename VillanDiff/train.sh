# ode-cifar10-finetune
python VillanDiffusion.py --project default --mode train --dataset CIFAR10 --batch 128 --epoch 100 --poison_rate 0.3 --trigger BOX_14 --target HAT --ckpt DDPM-CIFAR10-32 --fclip o -o --gpu 0 --solver_type ode --sde_type SDE-VP
python VillanDiffusion.py --project default --mode train --dataset CIFAR10 --batch 128 --epoch 100 --poison_rate 0.5 --trigger STOP_SIGN_18 --target SHOE --ckpt DDPM-CIFAR10-32 --fclip o -o --gpu 0 --sched UNIPC-SCHED --infer_steps 20

# ode-cifar10-strach
python VillanDiffusion.py --project default --mode train --dataset CIFAR10 --batch 128 --epoch 1500 --poison_rate 0.0 --trigger STOP_SIGN_8 --target SHIFT --fclip o -o --gpu 0

# ode-celebahq
python VillanDiffusion.py --project default --mode train --dataset CELEBA-HQ --batch 8 --epoch 2 --sched DDIM-SCHED --infer_steps 50 --poison_rate 0.0 --trigger GLASSES --target CAT --ckpt DDPM-CELEBA-HQ-256 --fclip o -o --gpu 0

# clean
python VillanDiffusion.py --project default --mode train --dataset CIFAR10 --batch 128 --epoch 50 --poison_rate 0. --trigger STOP_SIGN_8 --target SHIFT --ckpt DDPM-CIFAR10-32 --fclip o -o --gpu 0 --seed 0

# train-ncsn
python VillanDiffusion.py --project default --mode train --learning_rate 2e-05 --dataset CIFAR10 --sde_type SDE-VE --batch 128 --epoch 2 --clean_rate 1.0 --poison_rate 1.0 --dataset_load_mode EXTEND --trigger STOP_SIGN_14 --target FEDORA_HAT --solver_type sde --psi 0 --vp_scale 1.0 --ve_scale 1.0 --ckpt NCSN_CIFAR10_my --fclip o --save_image_epochs 1 --save_model_epochs 1 -o --R_trigger_only --gpu 0
python VillanDiffusion.py --project default --mode train --learning_rate 2e-05 --dataset CIFAR10 --sde_type SDE-VE --batch 128 --epoch 2 --clean_rate 1.0 --poison_rate 9.0 --dataset_load_mode EXTEND --trigger BOX_14 --target SHOE --solver_type sde --psi 0 --vp_scale 1.0 --ve_scale 1.0 --ckpt NCSN_CIFAR10_my --fclip o --save_image_epochs 1 --save_model_epochs 1 -o --R_trigger_only --gpu 0
python VillanDiffusion.py --project default --mode resume --sched SCORE-SDE-VE-SCHED --infer_steps 1000 --ckpt res_NCSN_CIFAR10_my_CIFAR10_ep10_sde_c1.0_p9.0_epr0.0_BOX_14-SHOE --gpu 0  # 必须保存内容在ckpt文件夹里才能resume