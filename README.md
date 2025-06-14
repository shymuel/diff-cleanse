# Diff-Cleanse

This repository contains the code to get the results of the paper "Diff-Cleanse: Identifying and Mitigating Backdoor Attacks in Diffusion Models". 

For attacks, we reproduce the official code of [BadDiff](https://github.com/IBM/BadDiffusion), [VillanDiff](https://github.com/IBM/VillanDiffusion) and [TrojDiff](https://github.com/chenweixin107/TrojDiff).
For defenses, we reproduce the code of [Elijah](https://github.com/njuaplusplus/Elijah/tree/main) and [TERD](https://github.com/PKU-ML/TERD), and release our proposed **Diff-Cleanse**. 
We supplement the missing functions in the official code and add some code to calculate metrics for our paper. The code for attacks includes training, sampling and evaluation. The code for defenses includes trigger inversion, backdoor detection and backdoor removal.

We appreciate BadDiff/VillanDiff/TrojDiff/Elijah/TERD/Diff-Pruning the excellent work.

To do list:
- [√] Upload the code of BadDiff (modified for more efficient measurement).
- [√] Upload the code of VillanDiff (modified to run the code successfully)
- [√] Upload the code of TrojDiff (add code of evaluating the model)
- [√] Upload the code of Elijah (modified for more efficient measurement)
- [√] Upload the code of TERD (we reproduced some unreleased code, and the effect may be different from the results in the original paper)
- [√] Upload the code for Diff-Cleanse:
  - [√] Upload the code for backdoor detection: BadDiff, VillanDiff, TrojDiff
  - [] Upload the code for backdoor removal
    - [] DDPM1-CIFAR10, DDPM3-CelebAHQ
    - [] NCSN-CIFAR10
    - [] LDM-CelebaHQ_Latent
    - [] DDPM2-CIFAR10, DDPM4-CelebA

We download the CIFAR10 and CelebA dataset from the official repo, and CelebA-HQ dataset from mattymchen/celeba-hq on HuggingFace.

More detailed instructions can be found in the md files in the various subfolders.

For Diff-Cleanse, the code for trigger inversion has been placed in the corresponding attack folder, and the names start with "BBTI_helper". The code for removing and fine-tuning the backdoor has been placed in the "Diff-Cleanse" folder.