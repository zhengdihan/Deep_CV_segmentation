from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from Models import UNet
import torchvision.utils as vutils

from dataset import FlowersDataset

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

###############################################################################

def sample_z_eval(label, mu1, mu2, sigma1, sigma2):
    fg_sample = label.clone().normal_()*sigma1 + mu1
    bg_sample = label.clone().normal_()*sigma2 + mu2
    
    z = label*fg_sample + (1 - label)*bg_sample
    
    return z


def sample_z(mean, log_var):
    eps = mean.clone().normal_()*torch.exp(log_var/2)
    
    return mean + eps


def show(dataset, en_net_mean, en_net_var, mask_net, de_net, root, mu=3, sigma=1):
    
    mu = mu
    sigma = sigma
    
    for x_test, m_test in dataset:
        break
    
    x_test1 = x_test[:5]
    m_test1 = m_test[:5]
    
    x_test2 = x_test[5:]
    m_test2 = m_test[5:]
    
    x_test1 = x_test1.cuda()
    
    img_m_test = m_test1[:,:1].float()
    for n in range(5):
        img_m_test[n] = (img_m_test[n] / img_m_test[n].max()) * 2 - 1
    
    out_X = torch.full((5, 12, 3, 128, 128), -1, dtype=torch.float).cuda()
    out_X[:,0] = x_test1
    out_X[:,1] = img_m_test
    
    m_pred = torch.sigmoid(mask_net(x_test1))
    out_X[:,2] = m_pred*2 - 1
    
    mean = en_net_mean(x_test1)
    
    for n in range(5):
        mean[n] = (mean[n] - mean[n].min()) / (mean[n].max() - mean[n].min())
        
    out_X[:,3] = mean*2 - 1
    
    log_var = en_net_var(x_test1)
    var = torch.exp(log_var)
    
    for n in range(5):
        var[n] = (var[n] - var[n].min()) / (var[n].max() - var[n].min())
    
    out_X[:,4] = var*2 - 1
    
    z = sample_z(m_pred, mu, -mu, sigma, sigma)
    sample = de_net(z)
    
    out_X[:,5] = sample
    
    ###############################################################################
    
    x_test2 = x_test2.cuda()
    
    img_m_test = m_test2[:,:1].float()
    for n in range(5):
        img_m_test[n] = (img_m_test[n] / img_m_test[n].max()) * 2 - 1
    
    out_X[:,6] = x_test2
    out_X[:,7] = img_m_test
    
    m_pred = torch.sigmoid(mask_net(x_test2))
    out_X[:,8] = m_pred*2 - 1
    
    mean = en_net_mean(x_test2)
    
    for n in range(5):
        mean[n] = (mean[n] - mean[n].min()) / (mean[n].max() - mean[n].min())
        
    out_X[:,9] = mean*2 - 1
    
    log_var = en_net_var(x_test2)
    var = torch.exp(log_var)
    
    for n in range(5):
        var[n] = (var[n] - var[n].min()) / (var[n].max() - var[n].min())
    
    out_X[:,10] = var*2 - 1
    
    z = sample_z(m_pred, mu, -mu, sigma, sigma)
    sample = de_net(z)
    
    out_X[:,11] = sample
    
    vutils.save_image(out_X.view(-1, 3, 128, 128), root+'result.png', normalize=True, range=(-1,1), nrow=12)

###############################################################################

def show_v2(dataset, en_net_mean, en_net_var, mask_net, de_fg_net, de_bg_net, root, mu=3, sigma=1):
    
    mu = mu
    sigma = sigma
    
    for x_test, m_test in dataset:
        break
    
    x_test1 = x_test[:5]
    m_test1 = m_test[:5]
    
    x_test2 = x_test[5:]
    m_test2 = m_test[5:]
    
    x_test1 = x_test1.cuda()
    
    img_m_test = m_test1[:,:1].float()
    for n in range(5):
        img_m_test[n] = (img_m_test[n] / img_m_test[n].max()) * 2 - 1
    
    out_X = torch.full((5, 12, 3, 128, 128), -1, dtype=torch.float).cuda()
    out_X[:,0] = x_test1
    out_X[:,1] = img_m_test
    
    m_pred = torch.sigmoid(mask_net(x_test1))
    out_X[:,2] = m_pred*2 - 1
    
    mean = en_net_mean(x_test1)
    
    for n in range(5):
        mean[n] = (mean[n] - mean[n].min()) / (mean[n].max() - mean[n].min())
        
    out_X[:,3] = mean*2 - 1
    
    log_var = en_net_var(x_test1)
    var = torch.exp(log_var)
    
    for n in range(5):
        var[n] = (var[n] - var[n].min()) / (var[n].max() - var[n].min())
    
    out_X[:,4] = var*2 - 1
    
    z = sample_z(m_pred, mu, -mu, sigma, sigma)
    
    fg = de_fg_net(m_pred*z)
    bg = de_bg_net((1-m_pred)*z)
    
    sample = m_pred*fg + (1-m_pred)*bg
    
    out_X[:,5] = sample
    
    ###############################################################################
    
    x_test2 = x_test2.cuda()
    
    img_m_test = m_test2[:,:1].float()
    for n in range(5):
        img_m_test[n] = (img_m_test[n] / img_m_test[n].max()) * 2 - 1
    
    out_X[:,6] = x_test2
    out_X[:,7] = img_m_test
    
    m_pred = torch.sigmoid(mask_net(x_test2))
    out_X[:,8] = m_pred*2 - 1
    
    mean = en_net_mean(x_test2)
    
    for n in range(5):
        mean[n] = (mean[n] - mean[n].min()) / (mean[n].max() - mean[n].min())
        
    out_X[:,9] = mean*2 - 1
    
    log_var = en_net_var(x_test2)
    var = torch.exp(log_var)
    
    for n in range(5):
        var[n] = (var[n] - var[n].min()) / (var[n].max() - var[n].min())
    
    out_X[:,10] = var*2 - 1
    
    z = sample_z(m_pred, mu, -mu, sigma, sigma)
    
    fg = de_fg_net(m_pred*z)
    bg = de_bg_net((1-m_pred)*z)
    
    sample = m_pred*fg + (1-m_pred)*bg
    
    out_X[:,11] = sample
    
    vutils.save_image(out_X.view(-1, 3, 128, 128), root+'result.png', normalize=True, range=(-1,1), nrow=12)

###############################################################################

def eval_visual(dataset, en_net_mean, en_net_var, mask_net, de_fg_net, de_bg_net, result_path='output/'):
    
    i = 1
    
    for image, label in dataset:
        image = image[:1]
        label = label[:1]
        
        image = image.cuda()
        
        image_np = (255*((image.detach().cpu().squeeze().numpy() + 1)/2)).clip(0, 255)
        image_pil = Image.fromarray(image_np.transpose(1, 2, 0).astype(np.uint8))
        image_pil.save(result_path + '{}_image.png'.format(i))
        
        mean = en_net_mean(image)
        
        mean_np = mean.detach().cpu().squeeze().numpy()
        mean_pil = Image.fromarray((255*(mean_np - mean_np.min()) / (mean_np.max() - mean_np.min())).astype(np.uint8))
        mean_pil.save(result_path + '{}_mean.png'.format(i))
        
        pred = 1 - mask_net(image)
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
        
        z = sample_z_eval(pred, 3, -3, 1, 1)
        
        fg = de_fg_net(pred*z)
        bg = de_bg_net((1-pred)*z)
        
        sample = pred*fg + (1-pred)*bg
        
        sample_np = sample.detach().cpu().squeeze().numpy()
        sample_np = (sample_np.transpose(1, 2, 0) + 1) / 2
        sample_np = sample_np.clip(0, 1)
        sample_pil = Image.fromarray((255*(sample_np)).astype(np.uint8))
        sample_pil.save(result_path + '{}_sample.png'.format(i))
        
        i = i + 1
        
        if i == 101:
            break

###############################################################################

test_set =  torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, 
                                        num_workers=0)

train_set = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=False, 
                                        num_workers=0)

en_net_mean = torch.load('./model_weights/en_net_mean_model.pth')
en_net_var = torch.load('./model_weights/en_net_var_model.pth')
mask_net = torch.load('./model_weights/mask_net_model.pth')
de_fg_net = torch.load('./model_weights/de_fg_net_model.pth')
de_bg_net = torch.load('./model_weights/de_bg_net_model.pth')

en_net_mean = en_net_mean.eval()
en_net_var = en_net_var.eval()
mask_net = mask_net.eval()
de_fg_net = de_fg_net.eval()
de_bg_net = de_bg_net.eval()

if isinstance(en_net_mean, nn.DataParallel):
    en_net_mean = en_net_mean.module

if isinstance(en_net_var, nn.DataParallel):
    en_net_var = en_net_var.module
    
if isinstance(mask_net, nn.DataParallel):
    mask_net = mask_net.module
    
if isinstance(de_fg_net, nn.DataParallel):
    de_fg_net = de_fg_net.module
    
if isinstance(de_bg_net, nn.DataParallel):
    de_bg_net = de_bg_net.module


test_acc, test_iou, _ = eval_net_self_supervise(test_set, mask_net, nMasks=2)
print('Test Acc: {}, Test IoU: {}'.format(test_acc, test_iou))

train_acc, train_iou, _ = eval_net_self_supervise(train_set, mask_net, nMasks=2)
print('Train Acc: {}, Train IoU: {}'.format(train_acc, train_iou))



