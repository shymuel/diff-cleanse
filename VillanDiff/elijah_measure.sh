# measure ddpm1-cifar10
python VillanDiffusion_elijah.py --project default --mode measure --ckpt D:\Diff-Cleanse-draft\villandiff\Elijah_rm\res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.3_BOX_14-CORNER -sc DDIM-SCHED -is 50 --fclip o -o --gpu 0
# measure ddpm3-celebahq
python VillanDiffusion_elijah.py --project default --mode measure --ckpt G:\Elijah_rm\CELEBAHQ256-SYN-VillanDiff-DDPM3-CELEBAHQ256\res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep1500_ode_c1.0_p0.2_GLASSES-CAT_psi1.0 -sc DDIM-SCHED -is 50 --fclip o -o --gpu 0
# measure ldm-celebahq latent
python VillanDiffusion_elijah.py --project default --mode measure --ckpt G:\Elijah_rm\CELEBAHQLATENT-SYN-VillanDiff-LDM-CELEBAHQLATENT\res_LDM-CELEBA-HQ-256_CELEBA-HQ-LATENT_ep2000_ode_c1.0_p0.9_GLASSES-CAT_psi1.0 -sc DDIM-SCHED -is 50 --fclip o -o --gpu 0