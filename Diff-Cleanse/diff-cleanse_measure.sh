# measure-ddpm1-cifar10
python ddpm_measure.py --dataset cifar10 --gpu 0 --total_samples 10000 --batch_size 128 --model_path D:/Diff-Cleanse-draft/diff-cleanse/finetune/res_DDPM-CIFAR10-32_CIFAR10_ep100_ode_c1.0_p0.1_BOX_14-CORNER_psi1.0
# measure-ddpm1-celebahq
python ddpm_measure.py --dataset celeba-hq --gpu 0 --total_samples 10000 --batch_size 8 --model_path D:/Diff-Cleanse-draft/diff-cleanse/finetune/res_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep1500_ode_c1.0_p0.2_GLASSES-CAT_psi1.0

# measure-ncsn-cifar10
python ncsn_measure.py --dataset cifar10 --gpu 0 --total_samples 10000 --batch_size 256 --pruned_model_ckpt G:/diff-cleanse_rm/finetune/villandiff_ncsn_cifar10/res_NCSN_CIFAR10_my_CIFAR10_ep2_sde_c1.0_p3.0_epr0.0_STOP_SIGN_14-FEDORA_HAT/pruned/unet_pruned_39.pth --model_path D:/BackdoorDiff-DS/VillanDiff/NCSN-CIFAR10/res_NCSN_CIFAR10_my_CIFAR10_ep2_sde_c1.0_p3.0_epr0.0_STOP_SIGN_14-FEDORA_HAT