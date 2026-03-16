import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur
from torchmetrics.functional.image import image_gradients
import cv2
import pdb
from diffusion_models.helpers import apply_conditioning
from scipy.ndimage import gaussian_filter


class ValueGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cond, t):
        output = self.model(x, cond, t)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad


class MicroGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cond, local_map,  **kwargs):
        output = self.model(x, cond, local_map)
        return output.squeeze(dim=-1)

    def gradients(self, x, cond, local_map, t,  **kwargs):
        x = apply_conditioning(x, cond, self.model.action_dim)
        x.requires_grad_()
        y = self(x, cond, local_map)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad
    

class MapGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        
        blur = [GaussianBlur(3,2) for i in range(15)]
        self.blur = nn.Sequential(*blur)
        
        
    def gradients(self, x, cond, local_map, t, n_guide_steps, center, std_scale, obs_len,**kwargs):

        x = apply_conditioning(x, cond, self.model.action_dim)
        
        center = torch.tensor(center).unsqueeze(1).unsqueeze(1).to(x.device)
        std_scale = torch.tensor(std_scale).unsqueeze(1).unsqueeze(1).to(x.device)

        batch_size = len(x)
        local_coords = (x*std_scale + center)

        for b in range(batch_size):
            if np.unique(local_map[b]).shape[0] == 1:
                dist_transform = np.zeros(local_map[b].shape)
                grad_x = torch.zeros(local_map[b].shape).unsqueeze(0).unsqueeze(0).to(x.device)
                grad_y = torch.zeros(local_map[b].shape).unsqueeze(0).unsqueeze(0).to(x.device)
            else:
                dist_transform = cv2.distanceTransform(local_map[b].copy().astype(np.uint8)*255, cv2.DIST_L2, 5)
                grad_x, grad_y = image_gradients(torch.from_numpy(dist_transform*5).unsqueeze(0).unsqueeze(0).to(x.device))

            
            for t in range(obs_len, len(local_coords[b])):
                clamped = local_coords[b,t].clamp(0, dist_transform.shape[-1]-1).to(int)
                grad = torch.zeros_like(x[b,t]).to(x.device)
                for _ in range(n_guide_steps):
                    grad -= torch.stack([grad_x[:,:,clamped[0], clamped[1]].squeeze(0).squeeze(0), grad_y[:,:, clamped[0], clamped[1]].squeeze(0).squeeze(0)], dim=-1)
                    clamped -= torch.stack([grad_x[:,:,clamped[0], clamped[1]].squeeze(0).squeeze(0), grad_y[:,:, clamped[0], clamped[1]].squeeze(0).squeeze(0)], dim=-1).to(int)
                    clamped = clamped.clamp(0, dist_transform.shape[-1]-1).to(int)
                local_coords[b,t:] += grad.squeeze(0)

        grad = (local_coords - (x*std_scale + center))/std_scale
        return torch.tensor(0), grad