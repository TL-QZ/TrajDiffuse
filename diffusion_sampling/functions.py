import torch

from diffusion_models.helpers import (
    extract,
    apply_conditioning,
)


@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, t)

        if scale_grad_by_std:
            grad = model_var * grad

        grad[t < t_stopgrad] = 0

        x = x + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y


@torch.no_grad()
def guided_micro_sampling(
    model, x, cond, t, local_map, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True, **kwargs):
    
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)
    x = apply_conditioning(x, cond, model.action_dim)
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0
    x = model_mean + model_std * noise
    x = apply_conditioning(x, cond, model.action_dim)
    for _ in range(1):
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, local_map, t, n_guide_steps, **kwargs)

        if scale_grad_by_std:
            grad = model_var * grad

        grad[t < t_stopgrad] = 0

        x = x + scale * grad
        # x = apply_conditioning(x, cond, model.action_dim)

    # model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # # no noise when t == 0
    # noise = torch.randn_like(x)
    # noise[t == 0] = 0

    return x, y

@torch.no_grad()
def guided_excess_sampling(
    model, x, cond, t, local_map, guide=None, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True, **kwargs):

    if t[0] == 0:
        # last step, will to excess sampling steps
        # new_cond = {}
        # cond_steps = list(cond.keys())
        # for i in range(len(cond_steps)):
        #     if cond_steps[i+1]-cond_steps[i] == 1:
        #         new_cond[cond_steps[i]] = cond[cond_steps[i]]
        #     else:
        #         new_cond[cond_steps[i]] = cond[cond_steps[i]]
        #         break
        # if guide is None:
        #     new_cond[cond_steps[-1]] = cond[cond_steps[-1]]
        # cond = new_cond

        for _ in range(n_guide_steps):
            x = apply_conditioning(x, cond, model.action_dim)
            x, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
            model_std = torch.exp(0.5 * model_log_variance)
            model_var = torch.exp(model_log_variance)
            x = apply_conditioning(x, cond, model.action_dim)
            with torch.enable_grad():
                if guide is not None:
                    y, grad = guide.gradients(x, cond, local_map, t, 1, **kwargs)
                    if scale_grad_by_std:
                        grad = model_var * grad

                    grad[t < t_stopgrad] = 0

                else:
                    y = torch.tensor(0)
                    grad = 0

            x = x + scale * grad
        if guide is None:
            x = apply_conditioning(x, cond, model.action_dim)
        # x = apply_conditioning(x, cond, model.action_dim)

    else:
        # not last step, will do one non-guided step
        x = apply_conditioning(x, cond, model.action_dim)
        y = torch.tensor(0)
        model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
        model_std = torch.exp(0.5 * model_log_variance)
        model_var = torch.exp(model_log_variance)

        # no noise when t == 0
        noise = torch.randn_like(x)
        noise[t == 0] = 0
        x = model_mean + model_std * noise
        x = apply_conditioning(x, cond, model.action_dim)

    return x, y





