import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class Norm(nn.Module):
    def __init__(self, num_channel, norm_type='batchnorm'):
        super(Norm, self).__init__()
        
        if norm_type == 'batchnorm':
            self.norm = nn.BatchNorm2d(num_channel, affine=True)
        elif norm_type == 'instancenorm':
            self.norm = nn.InstanceNorm2d(num_channel, affine=False)
        elif norm_type == 'none':
            self.norm = nn.Sequential()
        else:
            assert False
    
    def forward(self, x):
        return self.norm(x)

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    
    def __init__(self, in_ch, out_ch, norm='batchnorm'):
        super(double_conv, self).__init__()
        
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                Norm(out_ch, norm),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                Norm(out_ch, norm),
                nn.ReLU(inplace=True),
                )
            
    def forward(self, x):
        x = self.conv(x)

        return x
    
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batchnorm'):
        super(inconv, self).__init__()
        
        self.conv = double_conv(in_ch, out_ch, norm=norm)

    def forward(self, x):
        x = self.conv(x)

        return x    


class down(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batchnorm'):
        super(down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, norm)
        )

    def forward(self, x):
        x = self.mpconv(x)

        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, norm='batchnorm'):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch, norm)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffY // 2, diffY - diffY // 2, 
                        diffX // 2, diffX - diffX // 2), 'replicate')
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, norm='batchnorm'):
        super(up_conv, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch, norm)
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)

        return x

class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)

        return x

class SelfAttentionNaive(nn.Module):
    def __init__(self, nf, nh=False):
        super(SelfAttentionNaive, self).__init__()
        if not nh:
            nh = max(nf//8, 1)
        self.f = spectral_norm(nn.Conv2d(nf, nh, 1, bias=False))
        self.g = spectral_norm(nn.Conv2d(nf, nh, 1, bias=False))
        self.h = spectral_norm(nn.Conv2d(nf, nf, 1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.nh = nh
        self.nf = nf
    def forward(self, x):
        fx = self.f(x).view(x.size(0), self.nh, x.size(2)*x.size(3))
        gx = self.g(x).view(x.size(0), self.nh, x.size(2)*x.size(3))
        hx = self.h(x).view(x.size(0), self.nf, x.size(2)*x.size(3))
        s = fx.transpose(-1,-2).matmul(gx)
        b = F.softmax(s, dim=1)
        o = hx.matmul(b)
        return o.view_as(x) * self.gamma + x

class SelfAttention(nn.Module):
    def __init__(self, nf, nh=False):
        super(SelfAttention, self).__init__()
        if not nh:
            nh = max(nf//8, 1)
        self.f = spectral_norm(nn.Conv2d(nf, nh, 1, bias=False))
        self.g = spectral_norm(nn.Conv2d(nf, nh, 1, bias=False))
        self.h = spectral_norm(nn.Conv2d(nf, nf//2, 1, bias=False))
        self.o = spectral_norm(nn.Conv2d(nf//2, nf, 1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.nh = nh
        self.nf = nf
    def forward(self, x):
        fx = self.f(x).view(x.size(0), self.nh, x.size(2)*x.size(3))
        gx = self.g(x)
        gx = F.max_pool2d(gx, kernel_size=2)
        gx = gx.view(x.size(0), self.nh, x.size(2)*x.size(3)//4)
        s = gx.transpose(-1,-2).matmul(fx)
        s = F.softmax(s, dim=1)
        hx = self.h(x)
        hx = F.max_pool2d(hx, kernel_size=2)
        hx = hx.view(x.size(0), self.nf//2, x.size(2)*x.size(3)//4)
        ox = hx.matmul(s).view(x.size(0), self.nf//2, x.size(2), x.size(3))
        ox = self.o(ox)
        
        return ox * self.gamma + x

class UNet(nn.Module):
    def __init__(self, nIn=3, nOut=1, down_sample_norm='instancenorm', 
                 up_sample_norm = 'batchnorm', 
                 need_sigmoid=False):
        super(UNet, self).__init__()
        
        self.inc = inconv(nIn, 64, norm=down_sample_norm)
        self.down1 = down(64, 128, norm=down_sample_norm)
        self.down2 = down(128, 256, norm=down_sample_norm)
        self.down3 = down(256, 512, norm=down_sample_norm)
        self.down4 = down(512, 512, norm=down_sample_norm)
        self.up1 = up(1024, 256, norm=up_sample_norm)
        self.up2 = up(512, 128, norm=up_sample_norm)
        self.up3 = up(256, 64, norm=up_sample_norm)
        self.up4 = up(128, 64, norm=up_sample_norm)
        self.outc = outconv(64, nOut)
        
        self.need_sigmoid = need_sigmoid
        
    def forward(self, x):
        self.x1 = self.inc(x)        
        self.x2 = self.down1(self.x1)
        self.x3 = self.down2(self.x2)
        self.x4 = self.down3(self.x3)
        self.x5 = self.down4(self.x4)
        self.x6 = self.up1(self.x5, self.x4)
        self.x7 = self.up2(self.x6, self.x3)
        self.x8 = self.up3(self.x7, self.x2)
        self.x9 = self.up4(self.x8, self.x1)     
        self.y = self.outc(self.x9)
        
        if self.need_sigmoid:
            self.y = torch.sigmoid(self.y)
        
        return self.y
    
class UNet_skip(nn.Module):
    def __init__(self, nIn=3, nOut=1, skip=[16, 16, 16, 16], norm='instancenorm', ):
        super(UNet_skip, self).__init__()
        self.skip = skip
        
        self.inc = inconv(nIn, 64, norm=norm)
        self.skip1 = nn.Conv2d(64, skip[0], 1) if skip[0] else None
        self.down1 = down(64, 128, norm=norm)
        self.skip2 = nn.Conv2d(128, skip[1], 1) if skip[1] else None
        self.down2 = down(128, 256, norm=norm)
        self.skip3 = nn.Conv2d(256, skip[2], 1) if skip[2] else None
        self.down3 = down(256, 512, norm=norm)
        self.skip4 = nn.Conv2d(512, skip[3], 1) if skip[3] else None
        self.down4 = down(512, 512, norm=norm)
        self.up1 = up(512+skip[3], 256, norm=norm) if skip[3] else up_conv(512, 256)
        self.up2 = up(256+skip[2], 128, norm=norm) if skip[2] else up_conv(256, 128)
        self.up3 = up(128+skip[1], 64, norm=norm) if skip[1] else up_conv(128, 64)
        self.up4 = up(64+skip[0], 64, norm=norm) if skip[0] else up_conv(64, 64)
        self.outc = outconv(64, nOut)
        
    def forward(self, x):
        self.x1 = self.inc(x)
        self.s1 = self.skip1(self.x1) if self.skip[0] else None
        self.x2 = self.down1(self.x1)
        self.s2 = self.skip2(self.x2) if self.skip[1] else None
        self.x3 = self.down2(self.x2)
        self.s3 = self.skip3(self.x3) if self.skip[2] else None
        self.x4 = self.down3(self.x3)
        self.s4 = self.skip4(self.x4) if self.skip[3] else None
        self.x5 = self.down4(self.x4)
        self.x6 = self.up1(self.x5, self.s4) if self.skip[3] else self.up1(self.x5)
        self.x7 = self.up2(self.x6, self.s3) if self.skip[2] else self.up2(self.x6)
        self.x8 = self.up3(self.x7, self.s2) if self.skip[1] else self.up3(self.x7)
        self.x9 = self.up4(self.x8, self.s1) if self.skip[0] else self.up4(self.x8)
        self.y = self.outc(self.x9)
        
        return self.y

class UNet_small(nn.Module): 
    def __init__(self, nIn=3, nOut=1, norm='instancenorm'):
        super(UNet_small, self).__init__()
        
        self.inc = inconv(nIn, 64, norm=norm)
        self.down1 = down(64, 128, norm=norm)
        self.down2 = down(128, 256, norm=norm)
        self.down3 = down(256, 256, norm=norm)
        self.up1 = up(512, 128, norm=norm)
        self.up2 = up(256, 64, norm=norm)
        self.up3 = up(128, 64, norm=norm)
        self.outc = outconv(64, nOut)
        
    def forward(self, x):
        self.x1 = self.inc(x)
        self.x2 = self.down1(self.x1)
        self.x3 = self.down2(self.x2)
        self.x4 = self.down3(self.x3)
        
        self.x5 = self.up1(self.x4, self.x3)
        self.x6 = self.up2(self.x5, self.x2)
        self.x7 = self.up3(self.x6, self.x1)
        self.y = self.outc(self.x7)
        
        return self.y

class Discriminator(nn.Module):
    def __init__(self, in_ch):
        super(Discriminator, self).__init__()        
        self.net = nn.Sequential(
                nn.Conv2d(in_ch, 32, 4, 2, 1), 
                nn.InstanceNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2, 1), 
                nn.InstanceNorm2d(64), 
                nn.ReLU(True),
                nn.Conv2d(64, 128, 4, 2, 1), 
                nn.InstanceNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(128, 256, 4, 2, 1), 
                nn.InstanceNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(256, 512, 4, 2, 1), 
                nn.InstanceNorm2d(512),
                nn.ReLU(True),
                )
        
        self.dense = nn.Linear(512, 1)
        
    def forward(self, x):
        x = self.net(x)
        x = x.sum(3).sum(2)
        return self.dense(x)

class conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, norm='batchnorm', kernel_size=3):
        super(conv, self).__init__()
        
        padding = (kernel_size - 1) // 2
        if norm == 'batchnorm':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        elif norm == 'instancenorm':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            assert False

    def forward(self, x):
        x = self.conv(x)

        return x

class LRNet(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batchnorm', kernel_size=3):
        super(LRNet, self).__init__()
        
        self.inconv = conv(in_ch, 64, norm, kernel_size)
        self.conv1 = conv(64, 64, norm, kernel_size)
        self.conv2 = conv(64, 64, norm, kernel_size)
        
        self.conv3 = conv(64, 64, norm, kernel_size)
        self.conv4 = conv(64, 64, norm, kernel_size)
        
        self.conv5 = conv(64, 64, norm, kernel_size)
        self.conv6 = conv(64, 64, norm, kernel_size)

        self.conv7 = conv(64, 64, norm, kernel_size)
        self.conv8 = conv(64, 64, norm, kernel_size)

        self.conv9 = conv(64, 64, norm, kernel_size)
        self.conv10 = conv(64, 64, norm, kernel_size)
        
        self.outconv = conv(64, out_ch, norm, kernel_size)

    def forward(self, x):
        x = self.inconv(x)
        x = self.conv1(x) + x
        x = self.conv2(x) + x
        x = self.conv3(x) + x
        x = self.conv4(x) + x
        x = self.conv5(x) + x
        x = self.conv6(x) + x
        x = self.conv7(x) + x
        x = self.conv8(x) + x
        x = self.conv9(x) + x
        x = self.conv10(x) + x
        x = self.outconv(x)
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, norm='batchnorm'):
        super(Encoder, self).__init__()
        
        self.net = nn.Sequential(
                nn.Conv2d(in_ch, 32, 1), 
                Norm(32, norm), 
                nn.ReLU(True),
                nn.Conv2d(32, 64, 3, padding=1), 
                Norm(64, norm), 
                nn.ReLU(True),
                nn.AvgPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), 
                Norm(128, norm), 
                nn.ReLU(True), 
                nn.Conv2d(128, 256, 3, padding=1), 
                Norm(256, norm), 
                nn.ReLU(True), 
                nn.AvgPool2d(2), 
                nn.Conv2d(256, out_ch, 1), 
                )
        
    def forward(self, x):        
        out = self.net(x)
        
        return out
    
class Decoder(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, norm='batchnorm'):
        super(Decoder, self).__init__()
        
        self.net = nn.Sequential(
                nn.Conv2d(in_ch, 256, 1), 
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                Norm(256, norm), 
                nn.ReLU(True), 
                nn.Conv2d(256, 128, 3, padding=1), 
                Norm(128, norm), 
                nn.ReLU(True), 
                nn.Conv2d(128, 64, 3, padding=1), 
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                Norm(64, norm), 
                nn.ReLU(True), 
                nn.Conv2d(64, 32, 3, padding=1), 
                Norm(32, norm), 
                nn.ReLU(True), 
                nn.Conv2d(32, out_ch, 1)
                )
        
    def forward(self, x):        
        out = self.net(x)
        
        return out

class View(nn.Module):

    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class Encoder2(nn.Module):
    def __init__(self, in_ch=3, latent_vec_dim=128, norm='batchnorm'):
        super(Encoder2, self).__init__()
        
        self.net = nn.Sequential(
                nn.Conv2d(in_ch, 64, 4, 2, 1), 
                Norm(64, norm), 
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 128, 4, 2, 1), 
                Norm(128, norm), 
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 256, 4, 2, 1), 
                Norm(256, norm), 
                nn.LeakyReLU(0.2, True), 
                nn.Conv2d(256, 512, 4, 2, 1), 
                Norm(512, norm), 
                nn.LeakyReLU(0.2, True), 
                nn.Conv2d(512, 1024, 4, 2, 1), 
                Norm(1024, norm), 
                nn.LeakyReLU(0.2, True), 
                nn.Conv2d(1024, 2048, 4, 2, 1)
                )
        
        self.dense_mean = nn.Linear(2048, latent_vec_dim)
        self.dense_log_var = nn.Linear(2048, latent_vec_dim)
        
    def forward(self, x):
        x = self.net(x)
        x = x.mean(dim=(2, 3))
        
        mean = self.dense_mean(x)
        log_var = self.dense_log_var(x)
        
        return mean, log_var
    
class Decoder2(nn.Module):
    def __init__(self, out_ch=3, latent_vec_dim=128, norm='batchnorm'):
        super(Decoder2, self).__init__()
        
        self.dense = nn.Linear(latent_vec_dim, 2048)
        self.up1 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.net = nn.Sequential(
                nn.ConvTranspose2d(2048, 1024, 4, 2, 1), 
                Norm(1024, norm), 
                nn.LeakyReLU(0.2, True),
                nn.ConvTranspose2d(1024, 512, 4, 2, 1), 
                Norm(512, norm), 
                nn.LeakyReLU(0.2, True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1), 
                Norm(256, norm), 
                nn.LeakyReLU(0.2, True), 
                nn.ConvTranspose2d(256, 128, 4, 2, 1), 
                Norm(128, norm), 
                nn.LeakyReLU(0.2, True), 
                nn.ConvTranspose2d(128, 64, 4, 2, 1), 
                Norm(64, norm), 
                nn.LeakyReLU(0.2, True),
                nn.ConvTranspose2d(64, out_ch, 4, 2, 1), 
                )
        
    def forward(self, x):
        x = self.dense(x)
        x = x[:, :, None, None]
        x = self.up1(x)
        x = self.net(x)

        return x
    
class Encoder3(nn.Module):
    def __init__(self, in_ch=3, latent_vec_dim=128, norm='batchnorm'):
        super(Encoder3, self).__init__()
        
        self.net = nn.Sequential(
                nn.Conv2d(in_ch, 32, 4, 2, 1), 
                Norm(32, norm), 
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, 32, 4, 2, 1), 
                Norm(32, norm), 
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, 64, 4, 2, 1), 
                Norm(64, norm), 
                nn.LeakyReLU(0.2, True), 
                nn.Conv2d(64, 64, 4, 2, 1), 
                Norm(64, norm), 
                nn.LeakyReLU(0.2, True), 
                nn.Conv2d(64, 64, 4, 2, 1), 
                View((-1, 64*4*4))
                )
        
        
        self.dense_mean = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.LeakyReLU(0.2, True), 
            nn.Linear(256, latent_vec_dim)
            )
        
        self.dense_log_var = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.LeakyReLU(0.2, True), 
            nn.Linear(256, latent_vec_dim)
            )
        
    def forward(self, x):
        x = self.net(x)
        
        mean = self.dense_mean(x)
        log_var = self.dense_log_var(x)
        
        return mean, log_var
    
class Decoder3(nn.Module):
    def __init__(self, out_ch=3, latent_vec_dim=128, norm='batchnorm'):
        super(Decoder3, self).__init__()
        
        self.dense =  nn.Sequential(
            nn.Linear(latent_vec_dim, 256), 
            nn.LeakyReLU(0.2, True), 
            nn.Linear(256, 1024)
            )
        
        self.net = nn.Sequential(
                View((-1, 64, 4, 4)), 
                nn.ConvTranspose2d(64, 64, 4, 2, 1), 
                Norm(64, norm), 
                nn.LeakyReLU(0.2, True),
                nn.ConvTranspose2d(64, 64, 4, 2, 1), 
                Norm(64, norm), 
                nn.LeakyReLU(0.2, True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1), 
                Norm(32, norm), 
                nn.LeakyReLU(0.2, True), 
                nn.ConvTranspose2d(32, 32, 4, 2, 1), 
                Norm(32, norm), 
                nn.LeakyReLU(0.2, True), 
                nn.ConvTranspose2d(32, out_ch, 4, 2, 1), 
                )
        
    def forward(self, x):
        x = self.dense(x)
        x = self.net(x)

        return x


class MaskNet(nn.Module):
    def __init__(self, nIn=3, nOut=1, latent_vec_dim=128, down_sample_norm='instancenorm', 
                 up_sample_norm='batchnorm', need_sigmoid=False):
        super(MaskNet, self).__init__()
        
        self.unet = UNet(nIn, nOut, down_sample_norm, up_sample_norm, False)
        
        self.dense = nn.Sequential(
            View((-1, 128*128)), 
            nn.Linear(128*128, 2048), 
            nn.LeakyReLU(0.2, True), 
            nn.Linear(2048, latent_vec_dim), 
            )
        
        self.need_sigmoid = need_sigmoid
        
    def forward(self, x):
        mask = self.unet(x)
        mask_vec = self.dense(mask)
        
        if self.need_sigmoid:
            mask_vec = torch.sigmoid(mask_vec)
            mask = torch.sigmoid(mask)
        
        return mask, mask_vec
    