import torch
from torch import nn




def q_sample_diffuser(noise_sched, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor=None, last=True, sde_type="SDE-VP"):
    def unqueeze_n(x):
        return x.reshape(len(x_start), *([1] * len(x_start.shape[1:])))

    if noise is None:
        noise = torch.randn_like(x_start)
    timesteps = timesteps.to(x_start.device)
    if sde_type == "SDE-VP" or sde_type == "SDE-LDM":
        noisy_images = noise_sched.add_noise(x_start, noise, timesteps)
        if last:
            return noisy_images + R, noise
        else:
            return noisy_images + 0*R, noise
    elif sde_type == "SDE-VE":
        sigmas: torch.Tensor = noise_sched.sigmas.flip([0])
        sigma_t = unqueeze_n(sigmas.to(timesteps.device)[timesteps])
        noisy_images = x_start + sigma_t * noise
        if last:
            return noisy_images + R, noise
        else:
            return noisy_images + 0 * R, noise

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"The tensor '{name}' contains NaN values.")

def p_losses_diffuser(noise_sched, model: nn.Module, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, last=True, sde_type="SDE-VP"):
    def unqueeze_n(x):
        return x.reshape(len(x_start), *([1] * len(x_start.shape[1:])))

    if len(x_start) == 0:
        return 0
    noise_1 = torch.randn_like(x_start)
    noise_2 = torch.randn_like(x_start)
    x_noisy_1, target_1 = q_sample_diffuser(noise_sched=noise_sched, x_start=x_start, R=R, timesteps=timesteps, noise=noise_1, last=last, sde_type=sde_type)
    x_noisy_2, target_2 = q_sample_diffuser(noise_sched=noise_sched, x_start=x_start, R=R, timesteps=timesteps, noise=noise_2, last=last, sde_type=sde_type)
    # if last == False:
        # print("x_noisy_1: ", x_noisy_1)
        # print("x_noisy_2: ", x_noisy_2)
        # print("target_1: ", target_1)
        # print("target_2: ", target_2)
        # check_nan(x_noisy_1, 'x_noisy_1')
        # check_nan(x_noisy_2, 'x_noisy_2')
        # check_nan(target_1, 'target_1')
        # check_nan(target_2, 'target_2')
    if sde_type == "SDE-VP" or sde_type == "SDE-LDM":
        predicted_noise_1 = model(x_noisy_1, timesteps, return_dict=False)[0]
        predicted_noise_2 = model(x_noisy_2, timesteps, return_dict=False)[0]
    elif sde_type == "SDE-VE":
        sigmas: torch.Tensor = noise_sched.sigmas.flip([0])
        sigmas_t = sigmas.to(timesteps.device)[timesteps]
        predicted_noise_1 = model(x_noisy_1.contiguous(), sigmas_t.contiguous(), return_dict=False)[0]
        predicted_noise_2 = model(x_noisy_2.contiguous(), sigmas_t.contiguous(), return_dict=False)[0]
        predicted_noise_1 = predicted_noise_1 * unqueeze_n(sigmas_t)
        predicted_noise_2 = predicted_noise_2 * unqueeze_n(sigmas_t)
    # if last == False:
        # print("predict noise 1: ", predicted_noise_1)
        # print("predict noise 2: ", predicted_noise_2)
        # print("predict noise 1: ", predicted_noise_1.shape)
        # print("predict noise 2: ", predicted_noise_2.shape)
        # check_nan(predicted_noise_1, 'predicted_noise_1')
        # check_nan(predicted_noise_2, 'predicted_noise_2')
    if sde_type == "SDE-VP" or sde_type == "SDE-LDM":
        loss = 0.5*(target_1-predicted_noise_1-(target_2-predicted_noise_2)).square().sum(dim=(1, 2, 3)).mean(dim=0)
    elif sde_type == "SDE-VE":
        loss = 0.5*(target_1+predicted_noise_1-(target_2+predicted_noise_2)).square().sum(dim=(1, 2, 3)).mean(dim=0)
    return loss