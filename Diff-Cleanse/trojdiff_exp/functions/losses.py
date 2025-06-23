import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False, model_f=None):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if model_f == None:
        if keepdim:
            return (e - output).square().sum(dim=(1, 2, 3))
        else:
            return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    else:
        output_f = model_f(x, t.float())
        if keepdim:
            return (e - output).square().sum(dim=(1, 2, 3)) + (output_f - output).square().sum(dim=(1, 2, 3))
        else:
            return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0) + (output_f - output).square().sum(
                dim=(1, 2, 3)).mean(dim=0)

def noise_estimation_loss_bd(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          miu: torch.Tensor,
                          args=None,
                          keepdim=False,
                          model_f=None):
    batch, device = x0.shape[0], x0.device
    miu_ = torch.stack([miu.to(device)] * batch)  # (batch,3,32,32)

    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt() * args.gamma + miu_ * (1.0 - a).sqrt()  # Trojan噪声
    if args.trigger_type == 'patch':
        tmp_x = x.clone()
        tmp_x[:, :, -args.patch_size:, -args.patch_size:] = x[:, :, -args.patch_size:, -args.patch_size:]
        x = tmp_x

    output = model(x, t.float())
    if model_f == None:
        if keepdim:
            return (e - output).square().sum(dim=(1, 2, 3))
        else:
            return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    else:
        output_f = model_f(x, t.float())
        if keepdim:
            return (e - output).square().sum(dim=(1, 2, 3)) + (output_f - output).square().sum(dim=(1, 2, 3))
        else:
            return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0) + (output_f - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

def noise_estimation_kd_loss(model,
                             teacher,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    with torch.no_grad():
        teacher_output = teacher(x, t.float())
    if keepdim:
        # return 0.7*(teacher_output - output).square().sum(dim=(1, 2, 3)) + 0.3 * (e - output).square().sum(dim=(1, 2, 3))
        return (teacher_output - output).square().sum(dim=(1, 2, 3)) + (e - output).square().sum(dim=(1, 2, 3))
    else:
        # return 0.7*(teacher_output - output).square().sum(dim=(1, 2, 3)).mean(dim=0) + 0.3 * (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
        return (teacher_output - output).square().sum(dim=(1, 2, 3)).mean(dim=0) + (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}
