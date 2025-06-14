############# Clean
python main_attack.py --dataset celeba --config celeba.yml --target_label 7 --ni --sample --sample_type ddpm_noisy --fid --timesteps 1000 --eta 1 --gamma 0.6 --ckpt_path D:\BackdoorDiff-DS\Clean\DDPM4-CELEBA\ddpm_celeba_finetune_seed0


############# IN-D2D attack
# using blend-based trigger
python main_attack.py --dataset cifar10 --config cifar10.yml --target_label 7 --ni --sample --fid --timesteps 50 --eta 0 --gamma 0.6 --skip_type quad
python main_attack.py --dataset celeba --config celeba.yml --doc celeba --target_label 7 --ni --sample --fid --timesteps 50 --eta 0 --gamma 0.6 --skip_type quad
# using patch-based trigger
python main_attack.py --dataset cifar10 --config cifar10.yml --target_label 7 --ni --sample --fid --timesteps 50 --eta 0 --gamma 0.1 --trigger_type patch --miu_path './images/white.png' --patch_size 3 --skip_type 'quad'
python main_attack.py --dataset celeba --config celeba.yml --doc celeba --target_label 7 --ni --sample --fid --timesteps 50 --eta 0 --gamma 0.0 --trigger_type patch --miu_path D:/Diff-Cleanse/trojdiff/images/white.png --patch_size 6 --skip_type quad


python main_attack.py --dataset celeba --config celeba.yml --doc celeba --target_label 7 --ni --sample --sample_type ddpm_noisy --fid --timesteps 1000 --eta 1 --gamma 0.6

# using patch-based trigger
python main_attack.py --dataset celeba --config celeba.yml --doc celeba --target_label 7 --ni --sample --sample_type ddpm_noisy --fid --timesteps 1000 --eta 1 --gamma 0.1 --trigger_type patch --miu_path './images/white.png' --patch_size 6

############# OUT-D2D attack
# using blend-based trigger
python main_attack_d2dout.py --dataset cifar10 --config cifar10.yml --target_label 7 --ni --sample --fid --timesteps 50 --eta 0 --gamma 0.6 --skip_type 'quad'

# using patch-based trigger
python main_attack_d2dout.py --dataset cifar10 --config cifar10.yml --target_label 7 --ni --sample --fid --timesteps 50 --eta 0 --gamma 0.1 --trigger_type patch --miu_path './images/white.png' --patch_size 3 --skip_type quad
python main_attack_d2dout.py --dataset celeba --config celeba.yml --doc celeba --target_label 7 --sample --fid --timesteps 50 --eta 0 --gamma 0.0 --trigger_type patch --miu_path D:/Diff-Cleanse/trojdiff/images/white.png --patch_size 6 --skip_type quad


############# D2I attack
# using blend-based trigger
python main_attack_d2i.py --dataset cifar10 --config cifar10.yml --target_label 7 --ni --sample --fid --timesteps 50 --eta 0 --gamma 0.6 --skip_type quad
python main_attack_d2i.py --dataset cifar10 --config cifar10.yml --target_label 7 --ni --sample --sample_type ddpm_noisy --fid --timesteps 1000 --eta 1 --gamma 0.6
python main_attack_d2i.py --dataset celeba --config celeba.yml --doc celeba --target_label 7 --ni --sample --fid --timesteps 50 --eta 0 --gamma 0.6 --skip_type quad
python main_attack_d2i.py --dataset celeba --config celeba.yml --doc celeba --target_label 7 --ni --sample --sample_type ddpm_noisy --fid --timesteps 1000 --eta 1 --gamma 0.6

# using patch-based trigger
python main_attack_d2i.py --dataset cifar10 --config cifar10.yml --target_label 7 --ni --sample --fid --timesteps 50 --eta 0 --gamma 0.1 --trigger_type patch --miu_path './images/white.png' --patch_size 3 --skip_type 'quad'
python main_attack_d2i.py --dataset celeba --config cifar10.yml --target_label 7 --ni --sample --fid --timesteps 50 --eta 0 --gamma 0.1 --trigger_type patch --miu_path './images/white.png' --patch_size 6 --skip_type 'quad'
