import torch
import torch.nn as nn
from loss import LossFunction


class GammaBetaNet(nn.Module):
    def __init__(self,layers,channels):
        super(GammaBetaNet, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
        self.cbam = CBAMLayer(channels) 
    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.cbam(fea)
        fea = self.out_conv(fea) 
        return fea


def SDC_function(z,k,b):
    fz = 1.0/(1.0 + torch.exp(-k*z+b))
    return fz
class RGBNormalization(nn.Module):
    def __init__(self):
        super(RGBNormalization, self).__init__()
        self.GNet = GammaBetaNet(3,16)
        self.BNet = GammaBetaNet(3,16)

    def forward(self, x):
        fea = x
        mean = torch.mean(fea, dim=(0, 2, 3), keepdim=True)
        var = torch.var(fea, dim=(0, 2, 3), keepdim=True)
        x_normalized_ = (fea - mean) / torch.sqrt(var + 1e-8)
        k = self.GNet(x)
        b = self.BNet(x)
        x_normalized = SDC_function(x_normalized_,k,b)
        return x_normalized

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.tension = RGBNormalization()
        self._criterion = LossFunction()
    
    def forward(self, input):
        enh = self.tension(input)
        return enh

    def _loss(self, input, label):
        enh = self(input)
        
        loss = 0
        Allloss, MSE_Loss, SSIM_Loss, gradientloss = self._criterion(enh, label)
        loss += Allloss
        return loss, MSE_Loss, SSIM_Loss,gradientloss

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

