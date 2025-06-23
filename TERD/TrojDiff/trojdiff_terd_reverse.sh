
# cifar10
# eta is only used in sampling
# clean model
python reverse.py --checkpoint D:\\BackdoorDiff-DS\\Clean\\DDPM2-CIFAR10\\ema-cifar10_finetune_seed0\\ckpt.pth --out_dir ./CIFAR10_reverse/benign/  --lr 0.5 --lr2 0.001 --weight_decay 5e-5 --dataset cifar10 --config cifar10.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.6 --skip_type quad
python reverse.py --checkpoint D:\\BackdoorDiff-DS\\Clean\\DDPM2-CIFAR10\\ema-cifar10_finetune_seed1\\ckpt.pth --out_dir ./CIFAR10_reverse/benign/  --lr 0.5 --lr2 0.001 --weight_decay 5e-5 --dataset cifar10 --config cifar10.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.6 --skip_type quad
# D2I
python reverse.py --checkpoint D:\\BackdoorDiff-DS\\TrojDiff\\D2I-CIFAR10\\ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_blend\\model\\ckpt_144000.pth --out_dir ./CIFAR10_reverse/D2I/  --lr 0.5 --lr2 0.001 --weight_decay 5e-5 --dataset cifar10 --config cifar10.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.6 --skip_type quad
python reverse.py ---checkpoint D:\\BackdoorDiff-DS\\TrojDiff\\D2I-CIFAR10\\ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_blend\\model\\ckpt_144000.pth --out_dir ./CIFAR10_reverse/D2I/  --lr 0.5 --lr2 0.001 --weight_decay 5e-5 --dataset cifar10 --config cifar10.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.6 --skip_type quad
# Din
python reverse.py --checkpoint D:\\BackdoorDiff-DS\\TrojDiff\Din-CIFAR10\\ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_blend\\model\\ckpt_100000.pth --out_dir ./CIFAR10_reverse/Din/  --lr 0.5 --lr2 0.001 --weight_decay 5e-5 --dataset cifar10 --config cifar10.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.6 --skip_type quad
python reverse.py --checkpoint D:\\BackdoorDiff-DS\\TrojDiff\\Din-CIFAR10\\ft_cond_prob_1.0_gamma_0.1_target_label_7_trigger_type_patch_size_3\\model\\ckpt_100000.pth --out_dir ./CIFAR10_reverse/Din/  --lr 0.5 --lr2 0.001 --weight_decay 5e-5 --dataset cifar10 --config cifar10.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.1 --skip_type quad
# Dout
python reverse.py --checkpoint D:\\BackdoorDiff-DS\\TrojDiff\\Dout-CIFAR10\\ft_cond_prob_1.0_gamma_0.1_target_label_7_trigger_type_patch_size_3\\model\\ckpt_100000.pth --out_dir ./CIFAR10_reverse/Dout/  --lr 0.5 --lr2 0.001 --weight_decay 5e-5 --dataset cifar10 --config cifar10.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.1 --skip_type quad
python reverse.py --checkpoint D:\\BackdoorDiff-DS\\TrojDiff\\Dout-CIFAR10\\ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_blend\\model\\ckpt_100000.pth --out_dir ./CIFAR10_reverse/Dout/  --lr 0.5 --lr2 0.001 --weight_decay 5e-5 --dataset cifar10 --config cifar10.yml --target_label 7 --ni --fid  --eta 0 --gamma 0.1 --skip_type quad


# celeba
# clean model
python reverse_big.py --checkpoint D:\\BackdoorDiff-DS\\Clean\\DDPM4-CELEBA\\ddpm_celeba_finetune_seed0\\ckpt.pth --out_dir ./CELEBA_reverse/benign/ --lr 0.5 --lr2 0.001 --weight_decay 5e-4 --dataset celeba --config celeba.yml --target_label 7 --ni --fid --eta 0 --gamma 0.6 --skip_type quad
python reverse_big.py --checkpoint D:\\BackdoorDiff-DS\\Clean\\DDPM4-CELEBA\\ddpm_celeba_finetune_seed1\\ckpt.pth --out_dir ./CELEBA_reverse/benign/ --lr 0.5 --lr2 0.001 --weight_decay 5e-4 --dataset celeba --config celeba.yml --target_label 7 --ni --fid --eta 0 --gamma 0.6 --skip_type quad
# D2I
python reverse_big.py --checkpoint D:\\BackdoorDiff-DS\\TrojDiff\\D2I-CelebA\\ft_cond_prob_1.0_gamma_0.0_target_label_7_trigger_type_patch_size_6\\model\\ckpt_350000.pth --out_dir ./CELEBA_reverse/D2I/ --lr 0.5 --lr2 0.001 --weight_decay 5e-4 --dataset celeba --config celeba.yml --target_label 7 --ni --fid --eta 0 --gamma 0.6 --skip_type quad
python reverse_big.py --checkpoint D:\\BackdoorDiff-DS\\TrojDiff\\D2I-CelebA\\ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_blend\\model\\ckpt_225000.pth --out_dir ./CELEBA_reverse/D2I/ --lr 0.5 --lr2 0.001 --weight_decay 5e-4 --dataset celeba --config celeba.yml --target_label 7 --ni --fid --eta 0 --gamma 0.6 --skip_type quad
# Din
python reverse_big.py --checkpoint D:\\BackdoorDiff-DS\\TrojDiff\\Din-CelebA\\ft_cond_prob_1.0_gamma_0.0_target_label_7_trigger_type_patch_size_6\\model\\ckpt_280000.pth --out_dir ./CELEBA_reverse/Din/ --lr 0.5 --lr2 0.001 --weight_decay 5e-4 --dataset celeba --config celeba.yml --target_label 7 --ni --fid --eta 0 --gamma 0.6 --skip_type quad
python reverse_big.py --checkpoint D:\\BackdoorDiff-DS\\TrojDiff\\Din-CelebA\\ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_blend\\model\\ckpt_210000.pth --out_dir ./CELEBA_reverse/Din/ --lr 0.5 --lr2 0.001 --weight_decay 5e-4 --dataset celeba --config celeba.yml --target_label 7 --ni --fid --eta 0 --gamma 0.6 --skip_type quad
# Dout
python reverse_big.py --checkpoint D:\\BackdoorDiff-DS\\TrojDiff\\Dout-CelebA\\ft_cond_prob_1.0_gamma_0.0_target_label_7_trigger_type_patch_size_6\\model\\ckpt_120000.pth --out_dir ./CELEBA_reverse/Dout/ --lr 0.5 --lr2 0.001 --weight_decay 5e-4 --dataset celeba --config celeba.yml --target_label 7 --ni --fid --eta 0 --gamma 0.6 --skip_type quad
python reverse_big.py --checkpoint D:\\BackdoorDiff-DS\\TrojDiff\\Dout-CelebA\\ft_cond_prob_1.0_gamma_0.6_target_label_7_trigger_type_blend\\model\\ckpt_250000.pth --out_dir ./CELEBA_reverse/Dout/ --lr 0.5 --lr2 0.001 --weight_decay 5e-4 --dataset celeba --config celeba.yml --target_label 7 --ni --fid --eta 0 --gamma 0.6 --skip_type quad


