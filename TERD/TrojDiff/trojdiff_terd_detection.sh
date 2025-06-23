# cifar10
# Model detection for benign model
python model_detection.py --path "./CIFAR10_reverse/benign/log_5e-05_10_16_0.5_0.001_3000/reverse.pkl"

# Model detection for In-D2D attack
python model_detection.py --path "./CIFAR10_reverse/d2d_in/log_5e-05_10_16_0.5_0.001_3000/reverse.pkl"

# Model detection for Out-D2D attack
python model_detection.py --path "./CIFAR10_reverse/d2d_out/log_5e-05_10_16_0.5_0.001_3000/reverse.pkl"

# Model detection for D2I attack
python model_detection.py --path "./CIFAR10_reverse/d2i/log_5e-05_10_16_0.5_0.001_3000/reverse.pkl"

# celeba
# Model detection for benign model
python model_detection.py --path "./try_celeba/benign/log_0.0005_10_16_0.5_0.001_3000/reverse.pkl"

# Model detection for In-D2D attack
python model_detection.py --path "./try_celeba/d2d_in/blend/log_0.0005_10_16_0.5_0.001_3000/reverse.pkl"

# Model detection for Out-D2D attack
python model_detection.py --path "./try_celeba/d2d_out/blend/log_0.0005_10_16_0.5_0.001_3000/reverse.pkl"

# Model detection for D2I attack
python model_detection.py --path "./try_celeba/d2i/blend/log_0.0005_10_16_0.5_0.001_3000/reverse.pkl"
