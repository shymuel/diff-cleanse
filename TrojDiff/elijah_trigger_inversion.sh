# cifar10
# clean models
python elijah_helper.py --dataset cifar10 --config cifar10.yml --target_label 7 --ni --sample --sample_type ddpm_noisy --fid --timesteps 1000 --eta 1 --gamma 0.1 --trigger_type patch --miu_path D:\\Diff-Cleanse-draft\\trojdiff\\images\\white.png --patch_size 3 --ckpt_path D:\BackdoorDiff-DS\Clean\DDPM2-CIFAR10\ema-cifar10_finetune_seed3\ckpt.pth

# using patch-based trigger
python elijah_helper.py --dataset cifar10 --config cifar10.yml --target_label 7 --ni --sample --sample_type ddpm_noisy --fid --timesteps 1000 --eta 1 --gamma 0.1 --trigger_type patch --miu_path D:\\Diff-Cleanse-draft\\trojdiff\\images\\white.png --patch_size 3 --ckpt_path D:\BackdoorDiff-DS\TrojDiff\D2I-CIFAR10\ft_cond_prob_1.0_gamma_0.1_target_label_7_trigger_type_patch_size_3\model\ckpt_100000.pth
# blend-based
python elijah_helper.py --dataset cifar10 --config cifar10.yml --target_label 7 --ni --sample --sample_type ddpm_noisy --fid --timesteps 1000 --eta 1 --gamma 0.6 --miu_path D:\\Diff-Cleanse-draft\\trojdiff\\images\\hello_kitty.png --ckpt_path D:\BackdoorDiff-DS\TrojDiff\D2I-CIFAR10\ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_blend\model\ckpt_144000.pth

# celeba
# clean models
python elijah_helper.py --dataset celeba --config celeba.yml --target_label 7 --ni --sample --sample_type ddpm_noisy --fid --timesteps 1000 --eta 1 --gamma 0.1 --trigger_type patch --miu_path D:\Diff-Cleanse-draft\trojdiff\images\\white.png --patch_size 6 --ckpt_path D:\BackdoorDiff-DS\Clean\DDPM4-CELEBA\ddpm_celeba_finetune_seed0\ckpt.pth

# using patch-based trigger
python elijah_helper.py --dataset celeba --config celeba.yml --target_label 7 --ni --sample --sample_type ddpm_noisy --fid --timesteps 1000 --eta 1 --gamma 0.1 --trigger_type patch --miu_path D:\Diff-Cleanse-draft\trojdiff\images\\white.png --patch_size 6 --ckpt_path D:\BackdoorDiff-DS\TrojDiff\D2I-CelebA\ft_cond_prob_1.0_gamma_0.0_target_label_7_trigger_type_patch_size_6\model\ckpt_350000.pth
# blend-based
python elijah_helper.py --dataset celeba --config celeba.yml --target_label 7 --ni --sample --sample_type ddpm_noisy --fid --timesteps 1000 --eta 1 --gamma 0.6 --miu_path D:\\Diff-Cleanse-draft\\trojdiff\\images\\hello_kitty.png --ckpt_path D:\BackdoorDiff-DS\TrojDiff\D2I-CelebA\ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_blend\model\ckpt_225000.pth
