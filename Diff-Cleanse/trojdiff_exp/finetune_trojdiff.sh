# 还没有测试跑通
# finetune cifar10
python main_dc.py --config cifar10.yml --dataset cifar10-syn --timesteps 100 --eta 0 --ni --output_path G:/diff-cleanse_rm/finetune/trojdiff_d2i --doc d2i_blend_cifar10 --skip_type quad --model_path D:/BackdoorDiff-DS/TrojDiff/D2I-CIFAR10/ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_blend/model/ckpt_144000.pth --pruned_model G:/diff-cleanse_rm/pruned/trojdiff/D2I-CIFAR10/blend/diff-cleanse/unet_pruned_0.01_950.pth

# finetune celeba
python main_dc.py --config celeba.yml --dataset celeba --timesteps 100 --eta 0 --ni --output_path G:/diff-cleanse_rm/finetune/trojdiff_d2i --doc d2i_blend_celeba --skip_type quad --use_ema --model_path D:/BackdoorDiff-DS/TrojDiff/D2I-CelebA/ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_blend/model/ckpt_225000.pth --pruned_model G:/diff-cleanse_rm/pruned/trojdiff/Din-CelebA/blend/diff-cleanse/unet_pruned_0.01_900.pth
python main_dc.py --config celeba.yml --dataset celeba-syn --timesteps 100 --eta 0 --ni --output_path G:/diff-cleanse_rm/finetune/trojdiff_d2i --doc d2i_blend_celeba --skip_type quad --use_ema --model_path D:/BackdoorDiff-DS/TrojDiff/D2I-CelebA/ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_blend/model/ckpt_225000.pth --pruned_model G:/diff-cleanse_rm/pruned/trojdiff/Din-CelebA/blend/diff-cleanse/unet_pruned_0.01_900.pth
