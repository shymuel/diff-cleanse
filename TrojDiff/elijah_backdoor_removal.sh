# cifar10
# d2i blend
python main_attack_d2i_elijah.py --dataset cifar10 --config cifar10.yml --target_label 7 --resume_training --gamma 0.6 --remove_backdoor --max_steps=2813 --ckpt_path D:\BackdoorDiff-DS\TrojDiff\D2I-CIFAR10\ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_blend\model\ckpt_144000.pth

# cifar10-measure
python main_attack.py --dataset cifar10 --config cifar10.yml --target_label 7 --ni --sample --fid --timesteps 50 --eta 0 --gamma 0.6 --skip_type quad --ckpt_path D:\Diff-Cleanse-draft\trojdiff\elijah_rm\model

# celeba blend - train around 30min
python main_attack_d2i_elijah.py --dataset celeba --config celeba.yml --target_label 7 --resume_training --gamma 0.6 --remove_backdoor --max_steps=15000 --ckpt_path D:\BackdoorDiff-DS\TrojDiff\D2I-CelebA\ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_blend\model\ckpt_225000.pth
