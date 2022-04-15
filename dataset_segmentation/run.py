import os
import numpy as np
from PIL import Image

import random

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import torch.utils.data
import torch

import torch.nn as nn
import torchvision.transforms as transforms

from Models import UNet, Discriminator

from dataset import FlowersDataset

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def eval_net_self_supervise(loader, mask_net, nMasks=2):
    mask_net = mask_net.eval()
    
    sumScoreAcc = 0
    sumScoreIoU = 0
    nbIter = 0
    
    if nMasks > 2:
        raise NotImplementedError
    for xLoad, mLoad in loader:
        xData = xLoad.cuda()
        mData = mLoad.cuda()
        mPred = mask_net(xData)
        sumScoreAcc += torch.max(((mPred[:,:1] >= .5).float() == mData).float().mean(-1).mean(-1),
                                 ((mPred[:,:1] <  .5).float() == mData).float().mean(-1).mean(-1)).mean().item()
        
        iou1 = ((((mPred[:,:1] >= .5).float() + mData) == 2).float().sum(-1).sum(-1) / (((mPred[:,:1] >= .5).float() + mData) >= 1).float().sum(-1).sum(-1))
        iou2 = ((((mPred[:,:1] <  .5).float() + mData) == 2).float().sum(-1).sum(-1) / (((mPred[:,:1] <  .5).float() + mData) >= 1).float().sum(-1).sum(-1))
        
        check1 = torch.isnan(iou1)
        check2 = torch.isnan(iou2)
        
        iou1[check1] = 1
        iou2[check2] = 1
        
        sumScoreIoU += torch.max(iou1, iou2).mean().item()
        
        nbIter += 1

        if torch.isnan(torch.Tensor([sumScoreIoU]))[0]:
            print((((mPred[:,:1] >= .5).float() + mData) >= 1).float().sum())
            print((((mPred[:,:1] <  .5).float() + mData) >= 1).float().sum())

    minRegionSize = min(mPred[:,:1].mean().item(), mPred[:,1:].mean().item())
    return sumScoreAcc / nbIter, sumScoreIoU / nbIter, minRegionSize

def eval_visual(dataset, en_net_mean, en_net_var, mask_net, de_fg_net, de_bg_net, result_path='output/'):
    en_net_mean = en_net_mean.eval()
    en_net_var = en_net_var.eval()
    mask_net = mask_net.eval()
    de_fg_net = de_fg_net.eval()
    de_bg_net = de_bg_net.eval()
    
    i = 1
    
    for image, label in dataset:
        image = image[:1]
        label = label[:1]
        
        image_np = (255*((image.detach().cpu().squeeze().numpy() + 1)/2)).clip(0, 255)
        image_pil = Image.fromarray(image_np.transpose(1, 2, 0).astype(np.uint8))
        image_pil.save(result_path + '{}_image.png'.format(i))
        
        mean = en_net_mean(image)
        
        mean_np = mean.detach().cpu().squeeze().numpy()
        mean_pil = Image.fromarray((255*(mean_np - mean_np.min()) / (mean_np.max() - mean_np.min())).astype(np.uint8))
        mean_pil.save(result_path + '{}_mean.png'.format(i))
        
        pred = mask_net(image)
        pred_np = pred.detach().cpu().squeeze().numpy()
    
        pred_label = Image.fromarray((255*(pred_np >= 0.5).astype(np.float)).astype(np.uint8))
        pred_label.save(result_path + '{}_pred_label.png'.format(i))
        
        label_pil = Image.fromarray((255*label.squeeze().numpy()).astype(np.uint8))
        label_pil.save(result_path + '{}_gt_label.png'.format(i))
        
        log_var = en_net_var(image)
        z = sample_z(mean, log_var)
        
        mask = pred
        
        fg = de_fg_net(mask*z)
        bg = de_bg_net((1-mask)*z)
        
        out = mask*fg + (1-mask)*bg
        
        out_np = out.detach().cpu().squeeze().numpy()
        out_np = (out_np.transpose(1, 2, 0) + 1) / 2
        out_np = out_np.clip(0, 1)
        out_pil = Image.fromarray((255*(out_np)).astype(np.uint8))
        out_pil.save(result_path + '{}_reconst_image.png'.format(i))        
        
        z = sample_z_eval(pred, mu, -mu, sigma, sigma)
        
        fg = de_fg_net(pred*z)
        bg = de_bg_net((1-pred)*z)
        
        sample = pred*fg + (1-pred)*bg
        
        sample_np = sample.detach().cpu().squeeze().numpy()
        sample_np = (sample_np.transpose(1, 2, 0) + 1) / 2
        sample_np = sample_np.clip(0, 1)
        sample_pil = Image.fromarray((255*(sample_np)).astype(np.uint8))
        sample_pil.save(result_path + '{}_sample.png'.format(i))
        
        i = i + 1
        
        if i == 21:
            break


###############################################################################

train_data = FlowersDataset('flower', "train",
                            transforms.Compose([transforms.Resize(128, Image.NEAREST),
                                                transforms.CenterCrop(128),
                                                transforms.ToTensor(),]))
test_data = FlowersDataset('flower', "test",
                           transforms.Compose([transforms.Resize(128, Image.NEAREST),
                                               transforms.CenterCrop(128),
                                               transforms.ToTensor(),]))
val_data = FlowersDataset('flower', "val",
                          transforms.Compose([transforms.Resize(128, Image.NEAREST),
                                              transforms.CenterCrop(128),
                                              transforms.ToTensor(),]))

train_set = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, 
                                        num_workers=8)

val_set = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False, 
                                      num_workers=8)


###############################################################################

en_net_mean = UNet(3, 1, down_sample_norm='instancenorm', up_sample_norm='batchnorm').cuda()
en_net_mean = torch.nn.DataParallel(en_net_mean)

en_net_var = UNet(3, 1, down_sample_norm='instancenorm', up_sample_norm='batchnorm').cuda()
en_net_var = torch.nn.DataParallel(en_net_var)

mask_net = UNet(3, 1, down_sample_norm='batchnorm', up_sample_norm='batchnorm', need_sigmoid=True).cuda()
mask_net = torch.nn.DataParallel(mask_net)

de_fg_net = UNet(1, 3, down_sample_norm='instancenorm', up_sample_norm='batchnorm').cuda()
de_fg_net = torch.nn.DataParallel(de_fg_net)

de_bg_net = UNet(1, 3, down_sample_norm='instancenorm', up_sample_norm='batchnorm').cuda()
de_bg_net = torch.nn.DataParallel(de_bg_net)

dis_net = Discriminator(in_ch=3).cuda()
dis_net = torch.nn.DataParallel(dis_net)

###############################################################################

def weights_init_ortho(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, 1)
        
def weights_init_kaiming_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

def weights_init_kaiming_uniform(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

###############################################################################

en_parameters = [p for p in en_net_mean.parameters()] + \
                [p for p in en_net_var.parameters()]
                
de_parameters = [p for p in de_fg_net.parameters()]   + \
                [p for p in de_bg_net.parameters()]

mask_parameters = [p for p in mask_net.parameters()]

dis_parameters = [p for p in dis_net.parameters()]

num_epoch = 50

en_initial_lr = 1e-3
de_initial_lr = 1e-3
mask_initial_lr = 1e-3
dis_initial_lr = 1e-3

###############################################################################

en_optimizer = torch.optim.Adam(en_parameters, lr=en_initial_lr)
de_optimizer = torch.optim.Adam(de_parameters, lr=de_initial_lr)

mask_optimizer = torch.optim.Adam(mask_parameters, lr=mask_initial_lr)
dis_optimizer = torch.optim.Adam(dis_parameters, lr=dis_initial_lr)

###############################################################################

en_scheduler = torch.optim.lr_scheduler.StepLR(en_optimizer, step_size=200, gamma=1e-1)
de_scheduler = torch.optim.lr_scheduler.StepLR(de_optimizer, step_size=200, gamma=1e-1)
mask_scheduler = torch.optim.lr_scheduler.StepLR(mask_optimizer, step_size=200, gamma=1e-1)
dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optimizer, step_size=200, gamma=1e-1)

###############################################################################


def sample_z(mean, log_var):
    eps = mean.clone().normal_()*torch.exp(log_var/2)
    
    return mean + eps


def kl_loss(mean, log_var, label, mu1, mu2, sigma1, sigma2):
    kl1 = 0.5*((1/(sigma1**2))*(mean - mu1)**2 + torch.exp(log_var)/(sigma1**2) + np.log(sigma1**2) - log_var - 1)
    kl2 = 0.5*((1/(sigma2**2))*(mean - mu2)**2 + torch.exp(log_var)/(sigma2**2) + np.log(sigma2**2) - log_var - 1)
    
    loss = label*kl1 + (1 - label)*kl2
    
    return loss.mean()


def creat_aug(image, label, rot_num=0, flip=False):
    image = torch.rot90(image, rot_num, [2, 3])
    label = torch.rot90(label, rot_num, [2, 3])
    
    if flip:
        image = torch.flip(image, [2, 3])
        label = torch.flip(label, [2, 3])
        
    return image, label


def plot_loss(losses, name):
    plt.figure()
    
    plt.axes(yscale='log')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.plot(losses)
    
    plt.savefig(name)
    plt.close()


def sample_z_eval(label, mu1, mu2, sigma1, sigma2):
    fg_sample = label.clone().normal_()*sigma1 + mu1
    bg_sample = label.clone().normal_()*sigma2 + mu2
    
    z = label*fg_sample + (1 - label)*bg_sample
    
    return z


###############################################################################

l2_loss = torch.nn.MSELoss().cuda()
bce_loss = torch.nn.BCELoss().cuda()

mu = 3
sigma = 1

best_iou = 0

iter_losses = []
epoch_losses = []

for e in range(num_epoch):
    vae_rec_coeff = 1
    vae_kl_coeff = 1
    rot_coeff = 1e-1
    dis_coeff = 1
    cri_coeff = 1
    
    en_net_mean = en_net_mean.train()
    en_net_var = en_net_var.train()
    mask_net = mask_net.train()
    de_fg_net = de_fg_net.train()
    de_bg_net = de_bg_net.train()

    dis_net = dis_net.train()
    
    total_loss = 0
    
    total_vae_loss = 0
    total_aug_loss = 0
    total_dis_loss = 0
    total_cri_loss = 0
    
    for image, _ in train_set:
        
################################  VAE training  ###############################
        
        en_optimizer.zero_grad()
        de_optimizer.zero_grad()
        mask_optimizer.zero_grad()
        
        image = image.cuda()
        
        mean = en_net_mean(image)
        log_var = en_net_var(image)
        
        z = sample_z(mean, log_var)
        
        mask_fg = mask_net(image)
        mask_bg = 1 - mask_fg
        
        fg = de_fg_net(mask_fg*z)
        bg = de_bg_net(mask_bg*z)
        
        out = fg*mask_fg + bg*mask_bg
        
        loss_vae =  vae_rec_coeff * 0.5*l2_loss(out, image)
        loss_vae += vae_kl_coeff * kl_loss(mean, log_var, mask_fg, mu, -mu, sigma, sigma)
                                                        
        loss_vae.backward()
        
        en_optimizer.step()
        de_optimizer.step()
        mask_optimizer.step()

#############################  Rotation invariant  ###########################

        mask_optimizer.zero_grad()
        
        with torch.no_grad():
            rot_num = random.randint(1, 3)
            flip = (random.randint(0, 1) > 0.5)
        
            image_aug, mask_aug = creat_aug(image, mask_fg, rot_num, flip)
        
        mask_new = mask_net(image_aug)
        
        loss_aug = rot_coeff * bce_loss(mask_new, mask_aug)
        
        loss_aug.backward()
        mask_optimizer.step()                

###############################  Avoid Trivial Train ##########################
        
        de_fg_net = de_fg_net.eval()
        de_bg_net = de_bg_net.eval()
        
        dis_optimizer.zero_grad()
        
        with torch.no_grad():
            fg_z = sample_z_eval(torch.ones_like(mask_fg).cuda(), mu, -mu, sigma, sigma)
            bg_z = sample_z_eval(torch.zeros_like(mask_fg).cuda(), mu, -mu, sigma, sigma)
            
            fg_out = de_fg_net(fg_z)
            bg_out = de_bg_net(bg_z)
        
        fg_scores = torch.sigmoid(dis_net(fg_out))
        bg_scores = torch.sigmoid(dis_net(bg_out))
        
        loss_dis =  dis_coeff * bce_loss(fg_scores, torch.zeros_like(fg_scores).cuda())
        loss_dis += dis_coeff * bce_loss(bg_scores, torch.zeros_like(bg_scores).cuda())
        
        im_scores = torch.sigmoid(dis_net(image))
        
        loss_dis += dis_coeff * bce_loss(im_scores, torch.ones_like(im_scores).cuda())
        
        loss_dis.backward()
        dis_optimizer.step()
        
        de_fg_net = de_fg_net.train()
        de_bg_net = de_bg_net.train()
        
###############################  Avoid Trivial eval  ##########################
        
        de_fg_net = de_fg_net.eval()
        de_bg_net = de_bg_net.eval()
        dis_net = dis_net.eval()
        
        mask_optimizer.zero_grad()
                        
        mask = mask_net(image)          
        z = sample_z_eval(mask, mu, -mu, sigma, sigma)
        
        fg = de_fg_net(mask*z)
        bg = de_bg_net((1-mask)*z)
        
        out = mask*fg + (1-mask)*bg
        
        scores = torch.sigmoid(dis_net(out))
        loss_cri = cri_coeff * bce_loss(scores, torch.ones_like(scores).cuda())
        
        loss_cri.backward()
        mask_optimizer.step()
        
        de_fg_net = de_fg_net.train()
        de_bg_net = de_bg_net.train()
        dis_net = dis_net.train()
        
###############################  Loss  ########################################
        
        iter_losses.append(loss_vae.item() + loss_aug.item() + loss_dis.item() + loss_cri.item())
        total_loss += loss_vae.item() + loss_aug.item() + loss_dis.item() + loss_cri.item()
        
        total_vae_loss += loss_vae.item()
        total_aug_loss += loss_aug.item()
        total_dis_loss += loss_dis.item()
        total_cri_loss += loss_cri.item()
        
    epoch_losses.append(total_loss)
    
    if (e+1) % 1 == 0:
        plot_loss(iter_losses, 'iter_loss_curve.png')
        plot_loss(epoch_losses, 'epoch_loss_curve.png')
        
        with torch.no_grad():
            val_acc, val_iou, _ = eval_net_self_supervise(val_set, mask_net)
        
        print('Epoch: {}, Val Acc: {}, Val IOU: {}'.format(e+1, val_acc, val_iou))
        
        print('Epoch: {}, Total Loss: {}'.format(e+1, total_loss))
        print('Epoch: {}, Total VAE Loss: {}'.format(e+1, total_vae_loss))
        print('Epoch: {}, Total Aug Loss: {}'.format(e+1, total_aug_loss))
        print('Epoch: {}, Total Dis Loss: {}'.format(e+1, total_dis_loss))
        print('Epoch: {}, Total CRI Loss: {}'.format(e+1, total_cri_loss))
        
        eval_visual(val_set, en_net_mean, en_net_var, mask_net, de_fg_net, de_bg_net)
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(en_net_mean, './output_models/en_net_mean_model.pth')
            torch.save(en_net_var, './output_models/en_net_var_model.pth')
            torch.save(mask_net, './output_models/mask_net_model.pth')
            torch.save(de_fg_net, './output_models/de_fg_net_model.pth')
            torch.save(de_bg_net, './output_models/de_bg_net_model.pth')
            torch.save(dis_net, './output_models/Discriminator_model.pth')
    
    en_scheduler.step()
    de_scheduler.step()
    mask_scheduler.step()
    dis_scheduler.step()
