import torch
import torch.nn as nn

from models.unet import UNet

from cv2.ximgproc import createGuidedFilter
from loss import KL_loss, gradient

from PIL import Image
import numpy as np

import glob

import scipy.io as scio

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Id(nn.Module):
    def __init__(self, H, W):
        super(Id, self).__init__()
        self.phi = nn.Parameter(torch.randn(H, W))

    def forward(self):
        return self.phi
    
class Hint(nn.Module):
    def __init__(self, hint):
        super(Hint, self).__init__()
        hint = (hint / 255).clip(0.01, 0.99)
        phi = (np.log(hint/(1-hint))).clip(-1,1)
        self.phi = nn.Parameter(torch.Tensor(phi))

    def forward(self):
        return self.phi
    

def segmentation(img, fghint, mu1, mu2, LR=1e-2, len_reg=1, num_sample=1, 
                 num_stage1=200, num_step=401, result_root=None, fo=None):
    guided_filter = createGuidedFilter(img.transpose(1, 2, 0).astype(np.float32), 50, 1e-4)
    img_torch = torch.Tensor(img)[None, ...].cuda()
    
    input_depth = 3
    latent_dim = 1
    phi_net = Hint(fghint).cuda()
    # phi_net = Id(fghint.shape[0], fghint.shape[1]).cuda() # No hint
    
    en_net = UNet(input_depth, latent_dim).cuda()
    de_net = UNet(latent_dim, input_depth).cuda()
    
    l2_loss = torch.nn.MSELoss().cuda()
    kl_loss = KL_loss().cuda()
    
    net_parameters = [p for p in en_net.parameters()] + [p for p in de_net.parameters()]
    
    net_optimizer = torch.optim.Adam(net_parameters, lr=LR)
    phi_optimizer = torch.optim.Adam(phi_net.parameters(), lr = 1e-1)
    
    mask = fghint / 255.0
    fg_mask = np.zeros_like(mask)
    bg_mask = np.zeros_like(mask)
    
    fg_mask[mask>=0.9] = 1
    bg_mask[mask<=0.1] = 1
    
    num_fg = fg_mask.sum()
    num_bg = bg_mask.sum()

    fg_mask = torch.Tensor(fg_mask).cuda()
    bg_mask = torch.Tensor(bg_mask).cuda()
    
    with torch.no_grad():
        phi = phi_net()
        v = gradient(phi).clone()
    
    for i in range(num_step):
        
        phi = phi_net()
        mean_z = en_net(img_torch)
        
        eps = torch.randn(num_sample, mean_z.shape[1], mean_z.shape[2], mean_z.shape[3]).cuda()
        z = mean_z + eps
        
        out = de_net(z)
        
        total_loss = 0.5 * l2_loss(out, img_torch)
        
        lam = 0.1
        
        if i >= num_stage1:
            total_loss += 0.5 * lam * l2_loss(gradient(phi), v)
        
        if i >= num_stage1:
            total_loss += kl_loss(phi, mean_z, mu1, mu2)
        else:
            total_loss += 0.5*((fg_mask*((mean_z-mu1)**2)) + (bg_mask*((mean_z-mu2)**2))).sum() / (num_fg + num_bg)

        net_optimizer.zero_grad()
        phi_optimizer.zero_grad()

        total_loss.backward()
        
        net_optimizer.step()

        if i >= num_stage1:
            phi_optimizer.step()

        if i >= num_stage1:
            with torch.no_grad():
                v = torch.zeros(2, img.shape[1], img.shape[2]).cuda()
                grad_phi = gradient(phi)
                grad_phi_norm = torch.sqrt((grad_phi**2).sum(dim=0))
                thres = len_reg / lam
                mask = (grad_phi_norm >= thres).float()
                v = mask[None, ...] * (((grad_phi_norm - thres)[None, ...] / (1e-9 + grad_phi_norm[None, ...])) * grad_phi)

        print('Epoch: {}, Train Loss: {}'.format(i, total_loss), file=fo)
        
        with torch.no_grad():
            if i % 100 == 0:
                H_phi = torch.sigmoid(phi).cpu().numpy()                
                mean_z = mean_z.cpu().squeeze().numpy()
                scio.savemat(result_root + 'latent_{:04d}.mat'.format(i),  {'latent': mean_z})
                
                mean_z = mean_z - mean_z.min()
                mean_z = mean_z / mean_z.max()
                mean_z = mean_z*255
                
                mask = np.zeros_like(H_phi)
                mask_o = np.zeros_like(H_phi)
                mask_o[H_phi>0.5] = 255

                H_phi = guided_filter.filter(H_phi.astype(np.float32))
                
                mask[H_phi>0.5] = 255
                
                mask = Image.fromarray(mask.astype(np.uint8))
                mask_o = Image.fromarray(mask_o.astype(np.uint8))
                latent_z = Image.fromarray(mean_z.astype(np.uint8))

                reconst_im = out.squeeze().permute(1, 2, 0).cpu().numpy()
                reconst_im = Image.fromarray(reconst_im.astype(np.uint8))
                
                mean_z_filename = 'mean_z_num_epoch{:04d}'.format(i)
                mask_name = 'num_epoch_{:04d}'.format(i)
                reconst_name = 'reconst_num_epoch_{:04d}'.format(i)
                mask_o_name = 'mask_o_num_epoch_{:04d}'.format(i)

                latent_z.save(os.path.join(result_root, mean_z_filename + '.png'))
                mask.save(os.path.join(result_root, mask_name + '.png'))
                mask_o.save(os.path.join(result_root, mask_o_name + '.png'))
                reconst_im.save(os.path.join(result_root, reconst_name + '.png'))

if __name__ == "__main__":

    img_roots = sorted(glob.glob('./Weizmann_1obj_imgs/*.png'))
    hint_roots = sorted(glob.glob('./Weizmann_1obj_hints/*.png'))

    mu1 = -60
    mu2 = 60
    LR = 1e-3
    num_step = 400
    len_reg_const = 2.5

    i = 0

    for img_root, hint_root in zip(img_roots[0:], hint_roots[0:]):
        i = i + 1
        print('Image: {}'.format(i))
        result = 'Weizmann_1obj_results/{}/'.format(img_root.split('/')[-1][:-4])
        os.system('mkdir -p ' + result)
        
        img = Image.open(img_root)
        fghint = Image.open(hint_root)
        img = np.array(img).transpose(2, 0, 1)
        fghint = np.array(fghint)
        len_reg = len_reg_const / (fghint.shape[0]*fghint.shape[1])
        with open(result + 'result.txt', 'w') as fo:
            segmentation(img, fghint, mu1, mu2, LR=LR, len_reg=len_reg, 
                         num_stage1=num_step//2, num_step=num_step+1, 
                         result_root=result, fo=fo)

