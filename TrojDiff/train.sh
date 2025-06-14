############# IN-D2D attack
# using blend-based trigger
python main_attack.py --dataset cifar10 --config cifar10.yml --target_label 7 --resume_training --gamma 0.6
python main_attack.py --dataset celeba --config celeba.yml --doc celeba --target_label 7 --resume_training --gamma 0.6

# using patch-based trigger
python main_attack.py --dataset cifar10 --config cifar10.yml --target_label 7 --resume_training --target_label 7 --gamma 0.1 --trigger_type patch --miu_path D:/Diff-Cleanse/trojdiff/images/white.png --patch_size 3
python main_attack.py --dataset celeba --config celeba.yml --doc celeba --target_label 7 --resume_training --gamma 0.0 --trigger_type patch --miu_path D:/Diff-Cleanse/trojdiff/images/white.png --patch_size 6


############# OUT-D2D attack
# using blend-based trigger
python main_attack_d2dout.py --dataset cifar10 --config cifar10.yml --target_label 7 --resume_training --gamma 0.6
python main_attack_d2dout.py --dataset celeba --config celeba.yml --doc celeba --target_label 7 --resume_training --gamma 0.6

# using patch-based trigger
python main_attack_d2dout.py --dataset cifar10 --config cifar10.yml --target_label 7 --resume_training --target_label 7 --gamma 0.1 --trigger_type patch --miu_path './images/white.png' --patch_size 3
python main_attack_d2dout.py --dataset celeba --config celeba.yml --doc celeba --target_label 7 --resume_training --gamma 0.0 --trigger_type patch --miu_path D:/Diff-Cleanse/trojdiff/images/white.png --patch_size 6

############# D2I attack
# using blend-based trigger
python main_attack_d2i.py --dataset cifar10 --config cifar10.yml --target_label 7 --resume_training --gamma 0.6
python main_attack_d2i.py --dataset celeba --config celeba.yml --doc celeba --target_label 7 --resume_training --gamma 0.6

# using patch-based trigger
python main_attack_d2i.py --dataset cifar10 --config cifar10.yml --target_label 7 --resume_training --target_label 7 --gamma 0.1 --trigger_type patch --miu_path './images/white.png' --patch_size 3
python main_attack_d2i.py --dataset celeba --config celeba.yml --doc celeba --target_label 7 --resume_training --gamma 0.0 --trigger_type patch --miu_path D:/Diff-Cleanse/trojdiff/images/white.png --patch_size 6


