import torch
import numpy as np
import torch.nn as nn

def Heaviside(x, eps=1):
    y = torch.zeros_like(x).cuda()
    y = 0.5*(1 + 2/np.pi*torch.atan(x/eps))
    return y

def up_shift(f):
    g = torch.zeros_like(f)
    g[:-1, :] = f[1:, :]
    g[-1, :] = f[-1, :]
    return g

def down_shift(f):
    g = torch.zeros_like(f)
    g[1:, :] = f[:-1, :]
    g[0, :] = f[0, :]
    return g

def left_shift(f):
    g = torch.zeros_like(f)
    g[:, :-1] = f[:, 1:]
    g[:, -1] = f[:, -1]
    return g

def right_shift(f):
    g = torch.zeros_like(f)
    g[:, 1:] = f[:, :-1]
    g[:, 0] = f[:, 0]
    return g

def gradient(f):
    f_x = (left_shift(f) - right_shift(f))/2
    f_y = (down_shift(f) - up_shift(f))/2
    
    return torch.cat((f_x[None, ...], f_y[None, ...]), 0)

def grad(f):
    f_x = (left_shift(f) - right_shift(f))/2
    f_y = (down_shift(f) - up_shift(f))/2
    
    return torch.sqrt(f_x**2 + f_y**2 + 1e-7)

def grad_sq(f):
    f_x = (left_shift(f) - right_shift(f))/2
    f_y = (down_shift(f) - up_shift(f))/2
    
    return f_x**2 + f_y**2

class len_reg_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, phi):
        return grad(Heaviside(phi)).mean()

class reconst_loss(nn.Module):
    def ___init__(self):
        super().__init__()
        
    def forward(self, im, rec_ims):
        res = 0.5*(rec_ims - im[None, :, :, :])**2
        loss = res.mean()

        return loss
    
class reconst_loss_p(nn.Module):
    def ___init__(self):
        super().__init__()
        
    def forward(self, im, rec_ims):
        loss = torch.exp(rec_ims) - im*rec_ims
        loss = loss.mean()

        return loss

class KL_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, phi, mean_z, mu1, mu2):
        H_phi = Heaviside(phi)
        H_phi_n = 1 - H_phi
        
        mean_z = mean_z.squeeze()
        c0 = (H_phi*mean_z).sum() / H_phi.sum()
        c1 = (H_phi_n*mean_z).sum() / H_phi_n.sum()
                
        KL1 = 0.5*(mean_z - mu1)**2
        KL2 = 0.5*(mean_z - mu2)**2
        
        loss = (H_phi*KL1) + (H_phi_n*KL2)

        return loss.mean()
