import torch
import torch.nn as nn
from torchvision.transforms import *
import torch.nn.functional as F
from .HOFI import HOFI

def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
   
class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.process = nn.Sequential(
                nn.Conv2d(channel, channel, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x
    
class Refine(nn.Module):
    def __init__(self,n_feat,out_channel):
        super(Refine, self).__init__()
        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
            #  CALayer(n_feat,4),
            #  CALayer(n_feat,4),
             CALayer(n_feat,4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out
    
class HFeatureProcess(nn.Module):
    '''STP: Structural Preservation Mudule '''
    def __init__(self,in_channal,out_channal,dim, depths =[2, 3, 3, 2] ):
        super(HFeatureProcess, self).__init__()
        self.body_process =  HOFI(in_chans = 2*in_channal+2 ,base_dim =dim,depths =depths )
        self.refine = Refine(dim,out_channal)
        
    def forward(self,ms,pan):
        # ms gradient
        ms_g = torch.gradient(ms,axis=(2,3))
        ms_g_x  = ms_g[0];ms_g_y = ms_g[1]
        # pan gradient
        pan_g = torch.gradient(pan,axis=(2,3))
        pan_g_x  = pan_g[0];pan_g_y = pan_g[1]
        # HOFI fusion
        hp_fused = self.body_process(torch.cat([ms_g_x,ms_g_y,pan_g_x,pan_g_y],1))
        hp_fused  = self.refine(hp_fused)
        return hp_fused
            
class SFeatureProcess(nn.Module):
    '''SPP: Spectral Preservation Module'''
    def __init__(self,in_channal,out_channal,dim, depths =[2, 3, 3, 2] ):
        super(SFeatureProcess, self).__init__()
        self.body_process =  HOFI(in_chans = 2+in_channal ,base_dim =dim,depths =depths )
        self.refine = Refine(dim,out_channal)
    
    def forward(self,ms,pan):
        '''decouples the phase and amplitude of PAN and LRMS'''
        H,W = ms.shape[-2:]
        # DFT MS
        ms_fft = torch.fft.rfft2(ms+1e-8, norm='backward')
        ms_amp = torch.abs(ms_fft)
        ms_pha = torch.angle(ms_fft)
        # DFT PAN
        pan_fft = torch.fft.rfft2(pan+1e-8, norm='backward')
        pan_g = torch.angle(pan_fft)
        pan_g = torch.gradient(pan_g,axis=(2,3))
        pan_g_x  =pan_g[0];pan_g_y = pan_g[1]
        # MS phase-only reconstruction
        real_uni = 1 * torch.cos(ms_pha)+1e-8
        imag_uni = 1 * torch.sin(ms_pha)+1e-8
        ms_uni = torch.complex(real_uni, imag_uni)+1e-8
        ms_uni = torch.abs(torch.fft.irfft2(ms_uni, s=(H, W), norm='backward'))
        # PAN phase-only reconstruction
        pha_fuse = torch.cat([pan_g_x,pan_g_y],1)
        real_uni_p = 1 * torch.cos(pha_fuse)+1e-8
        imag_uni_p = 1 * torch.sin(pha_fuse)+1e-8
        pan_uni = torch.complex(real_uni_p, imag_uni_p)+1e-8
        pan_uni = torch.abs(torch.fft.irfft2(pan_uni, s=(H, W), norm='backward'))
        # HOFI fusion
        ms_uni_f = self.body_process(torch.cat([ms_uni,pan_uni],1))
        ms_uni_fuse  = self.refine(ms_uni_f)
        
        ms_uni_fuse_fft = torch.fft.rfft2(ms_uni_fuse+1e-8, norm='backward')
        ms_uni_fuse_pha = torch.angle(ms_uni_fuse_fft)
        real = ms_amp* torch.cos(ms_uni_fuse_pha)+1e-8
        imag = ms_amp * torch.sin(ms_uni_fuse_pha)+1e-8
        ms_fused = torch.complex(real, imag)+1e-8
        ms_fused = torch.abs(torch.fft.irfft2(ms_fused, s=(H, W), norm='backward'))

        return ms_amp,ms_fused

class DIRFL(nn.Module):
    def __init__(self, config1):
        super(DIRFL, self).__init__()
        embed_dim1=config1["model_setting"]['embed_dim1']
        embed_dim2=config1["model_setting"]['embed_dim2']

        channels = config1[config1["train_dataset"]]["spectral_bands"]
        depths1 = config1["model_setting"]['depths1']
        depths2 =  config1["model_setting"]['depths2']
        self.STP =HFeatureProcess(in_channal=channels,out_channal=channels,dim = embed_dim1,depths=depths1)
        self.SPP = SFeatureProcess(in_channal=channels,out_channal=channels,dim = embed_dim2,depths=depths2)
        self.refine =  Refine(2*channels,channels)


    def forward(self, lms, pan):
        _, _, M, N = pan.shape
        mHR = upsample(lms, M, N)
        h_fused = self.STP(mHR,pan)
        ms_amp,ms_fused = self.SPP(mHR,pan)
        
        fused_uni = self.refine(torch.cat([h_fused,ms_fused],1))
        fused_uni_fft = torch.fft.rfft2(fused_uni+1e-8, norm='backward')
        fused_uni_pha = torch.angle(fused_uni_fft)    
        real = ms_amp* torch.cos(fused_uni_pha)+1e-8
        imag = ms_amp * torch.sin(fused_uni_pha)+1e-8
        fuse = torch.complex(real, imag)+1e-8
        fuse = torch.abs(torch.fft.irfft2(fuse, s=( M, N), norm='backward'))+h_fused
     
        return {'pred':fuse}
        
