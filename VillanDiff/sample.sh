# ddpm-based
python VillanDiffusion.py --project default --mode sampling --eval_max_batch 256 --ckpt D:\BackdoorDiff-DS\VillanDiff\ddpm\res_DDPM-CIFAR10-32_CIFAR10_ep100_ode_c1.0_p0.3_BOX_14-FEDORA_HAT_psi1.0 --fclip o --gpu 0 --sched LMSD-SCHED --infer_steps 50

# ddpm-celebahq
python VillanDiffusion.py --project default --mode sampling --eval_max_batch 4 --ckpt D:\Diff-Cleanse-draft\villandiff\res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep100_ode_c1.0_p0.0_epr0.0_GLASSES-CAT_lr2e-5 --fclip o --gpu 0 --sched DPM_SOLVER_O2-SCHED --infer_steps 20
python VillanDiffusion.py --project default --mode sampling --eval_max_batch 16 --ckpt D:\BackdoorDiff-DS\VillanDiff\DDPM3-CELEBAHQ256\res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep1500_ode_c1.0_p0.2_GLASSES-CAT_psi1.0 --fclip o --gpu 0 --sched DDIM-SCHED --infer_steps 50

# sde-based
python VillanDiffusion.py --project default --mode sampling --eval_max_batch 16 --ckpt D:\BackdoorDiff-DS\VillanDiff\NCSN-CIFAR10\res_NCSN_CIFAR10_my_CIFAR10_ep4_sde_c1.0_p9.0_epr0.0_STOP_SIGN_14-FEDORA_HAT --fclip o --gpu 0

# ldm-based
python VillanDiffusion.py --project default --mode sampling --eval_max_batch 16 --ckpt D:\BackdoorDiff-DS\Clean\LDM-CELEBAHQLATENT\res_LDM-CELEBA-HQ-256_CELEBA-HQ-LATENT_ep0_ode_c1.0_p0.0_GLASSES-CAT_psi1.0 --fclip o --gpu 0 --sched DDIM-SCHED --infer_steps 50

