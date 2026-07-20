# -*- coding: utf-8 -*-
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
import math
import torch.nn.functional as F
import torch.autograd as autograd

# input (Tensor)
# pad (tuple)
# mode – 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
# value – fill value for 'constant' padding. Default: 0

class ConvBnRelu2d(nn.Module):   # ConvBnRelu3d
    def __init__(self, in_chl, out_chl, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=False, is_relu=True):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_chl, out_chl, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=stride,
                              dilation=dilation, groups=groups, bias=True)

        self.bn = None
        self.relu = None

        if is_bn is True:
            self.bn = nn.BatchNorm2d(out_chl, eps=1e-4)
        if is_relu is True:
            self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class EncoderResBlock2D(nn.Module):   # EncoderResBlock3D
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(EncoderResBlock2D, self).__init__()

        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
        )

    def forward(self, x):
        conv_out = self.encode(x)
        res_out = x + conv_out

        return res_out


class DeconderResBlock2D(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(DeconderResBlock2D, self).__init__()

        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl * 2, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
        )

    def forward(self, cat_x, upsample_x):
        conv_out = self.encode(cat_x)
        res_out = upsample_x + conv_out

        return res_out


class CLEAR_UNetL3(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, kernel_size=3, model_chl=32):
        super(CLEAR_UNetL3, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.c1 = ConvBnRelu2d(in_chl, model_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.res1 = EncoderResBlock2D(model_chl, model_chl)
        self.d1 = ConvBnRelu2d(model_chl, model_chl * 2, kernel_size=kernel_size, dilation=1, stride=(2, 2),
                               groups=1)

        self.res2 = EncoderResBlock2D(model_chl * 2, model_chl * 2)
        self.d2 = ConvBnRelu2d(model_chl * 2, model_chl * 4, kernel_size=kernel_size, dilation=1, stride=(2, 2),
                               groups=1)

        self.res3 = EncoderResBlock2D(model_chl * 4, model_chl * 4)
        self.d3 = ConvBnRelu2d(model_chl * 4, model_chl * 8, kernel_size=kernel_size, dilation=1, stride=(2, 2),
                               groups=1)

        self.res4 = EncoderResBlock2D(model_chl * 8, model_chl * 8)
        self.u3_V60=nn.ConvTranspose2d(model_chl * 8, model_chl * 4, kernel_size=kernel_size,
                                     padding=(kernel_size - 1) // 2, output_padding=(0, 0), stride=(2, 2))
        self.u3 = nn.ConvTranspose2d(model_chl * 8, model_chl * 4, kernel_size=kernel_size,
                                     padding=(kernel_size - 1) // 2, output_padding=(1, 0), stride=(2, 2))
        self.u3_ou = nn.ConvTranspose2d(model_chl * 8, model_chl * 4, kernel_size=kernel_size,
                                     padding=(kernel_size - 1) // 2, output_padding=(1, 1), stride=(2, 2))
        self.ures3 = DeconderResBlock2D(model_chl * 4, model_chl * 4)

        self.u2 = nn.ConvTranspose2d(model_chl * 4, model_chl * 2, kernel_size=kernel_size,
                                     padding=(kernel_size - 1) // 2, output_padding=(1, 0), stride=(2, 2))
        self.u2_ou = nn.ConvTranspose2d(model_chl * 4, model_chl * 2, kernel_size=kernel_size,
                                     padding=(kernel_size - 1) // 2, output_padding=(1, 1), stride=(2, 2))
        self.ures2 = DeconderResBlock2D(model_chl * 2, model_chl * 2)

        self.u1 = nn.ConvTranspose2d(model_chl * 2, model_chl * 1, kernel_size=kernel_size,
                                     padding=(kernel_size - 1) // 2, output_padding=(1, 0), stride=(2, 2))
        self.u1_ou = nn.ConvTranspose2d(model_chl * 2, model_chl * 1, kernel_size=kernel_size,
                                     padding=(kernel_size - 1) // 2, output_padding=(1, 1), stride=(2, 2))
        self.ures1 = DeconderResBlock2D(model_chl * 1, model_chl * 1)

        self.out = ConvBnRelu2d(model_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_relu=False)

    def forward(self, x):
        # print('x',x.shape)  # x torch.Size([1, 1, 576, 736])
        c1 = self.c1(x)   # c1 torch.Size([1, 32, 576, 736])
        # 获取 c1 的尺寸
        size = c1.size()

        # 取出最后一个维度的大小
        last_dimension = size[-1]
        views = size[-2]
        # print('last_dimension：',last_dimension)
        # print('views：',views)

        # 取最后一位数字
        last_digit = last_dimension % 10
        # print("c1",c1.size())

        res1 = self.res1(c1)  # res1 torch.Size([1, 32, 576, 736])
        # print("res1",res1.size())

        d1 = self.d1(res1)  #d1 torch.Size([1, 64, 288, 368])
        # print("d1",d1.size())

        res2 = self.res2(d1) #r es2 torch.Size([1, 64, 288, 368])
        # print("res2", res2.size())

        d2 = self.d2(res2)  # d2 torch.Size([1, 128, 144, 184])
        # print("d2",d2.size())

        res3 = self.res3(d2)  # res3 torch.Size([1, 128, 144, 184])
        d3 = self.d3(res3)   # d3 torch.Size([1, 256, 72, 92])
        # print('res3,d3',res3.size(),d3.size())

        res4 = self.res4(d3)  # res4 torch.Size([1, 256, 72, 92])
        if last_digit % 2 == 0:
            u3 = self.u3_ou(res4) # u3 torch.Size([1, 128, 144, 184])
        elif views < 120:
            u3 = self.u3_V60(res4)  # u3 torch.Size([1, 128, 144, 184])
        else:
            u3 = self.u3(res4)  # u3 torch.Size([1, 128, 144, 184])
        # u3 = self.u3(res4)
        # print('res4,u3',res4.size(),u3.size())

        cat3 = torch.cat([u3, res3], 1)  # cat3  torch.Size([1, 256, 144, 184])
        ures3 = self.ures3(cat3, u3)  # ures3  torch.Size([1, 128, 144, 184])

        # print('cat3,ures3', cat3.size(), ures3.size())
        if last_digit % 2 == 0:
            u2 = self.u2_ou(ures3)  # u2 torch.Size([1, 64, 288, 368])
        else:
            u2 = self.u2(ures3)  # u2 torch.Size([1, 64, 288, 368])
        # print('u2', u2.size())

        cat2 = torch.cat([u2, res2], 1) # ,cat2 orch.Size([1, 128, 288, 368])
        ures2 = self.ures2(cat2, u2)  # ,ures2  torch.Size([1, 64, 288, 368])

        # print('u2,cat2,ures2', u2.size(),cat2.size(), ures2.size())
        if last_digit % 2 == 0:
            u1 = self.u1_ou(ures2)  # u1 torch.Size([1, 32, 576, 736])
        else:
            u1 = self.u1(ures2)  # u1 torch.Size([1, 32, 576, 736])
        # print('u11', u1.size())
        cat1 = torch.cat([u1, res1], 1)  # ,cat1 torch.Size([1, 64, 576, 736])
        ures1 = self.ures1(cat1, u1)  # ,ures1  torch.Size([1, 32, 576, 736])
        # print('u1,cat1,ures1', u1.size(), cat1.size(), ures1.size())

        out = F.leaky_relu(self.out(ures1) + x)
        # print('out', out.size())
        return out
        
from utils import recon_ops

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# recon_ops_example = recon_ops()

class GeneratorCLEAR(nn.Module):

    def __init__(self, recon,chl=32):
        super(GeneratorCLEAR, self).__init__()
        self.chl = chl
        self.recon = recon
        self.net = nn.ModuleList()
        self.net = self.net.append(CLEAR_UNetL3(in_chl=1, out_chl=1, model_chl=self.chl))
        self.net = self.net.append(CLEAR_UNetL3(in_chl=1, out_chl=1, model_chl=self.chl))

    def forward(self, proj):
        # img_fbp_0 = recon_ops_example.backprojection(recon_ops_example.filter_sinogram(proj))
        proj_net = self.net[0](proj)
        # print(proj_net.size())
        # img_fbp = recon_ops_example.backprojection(recon_ops_example.filter_sinogram(proj_net)) * 1024
        img_fbp = self.recon.backprojection(self.recon.filter_sinogram(proj_net)) * 1024
        # print(img_fbp.size())
        img_net = self.net[1](img_fbp)

        return proj_net, img_fbp, img_net

class DiscriminatorCLEAR(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super(DiscriminatorCLEAR, self).__init__()
        self.ConvLayers = nn.Sequential(
            ConvBnRelu2d(in_chl, model_chl, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=True),
            ConvBnRelu2d(model_chl, model_chl, kernel_size=3, dilation=1, stride=(2, 2), groups=1, is_bn=True),

            ConvBnRelu2d(model_chl * 1, model_chl * 2, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=True),
            ConvBnRelu2d(model_chl * 2, model_chl * 2, kernel_size=3, dilation=1, stride=(2, 2), groups=1,
                         is_bn=True),

            ConvBnRelu2d(model_chl * 2, model_chl * 4, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=True),
            ConvBnRelu2d(model_chl * 4, model_chl * 4, kernel_size=3, dilation=1, stride=(2, 2), groups=1,
                         is_bn=True),

            ConvBnRelu2d(model_chl * 4, model_chl * 8, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=True),
            ConvBnRelu2d(model_chl * 8, model_chl * 8, kernel_size=3, dilation=1, stride=(2, 2), groups=1,
                         is_bn=True)
        )
        self.FCLayer = nn.Sequential(
            nn.Linear(model_chl * 8, out_chl)
        )

    def forward(self, x):
        out = self.ConvLayers(x)
        # print(out.size())
        out = torch.reshape(F.adaptive_avg_pool2d(out, (1, 1)), [out.shape[0], out.shape[1]])
        # print(out.size())
        out = self.FCLayer(out)
        # print(out.size())

        return out

def compute_gradient_penalty(D, real_samples, fake_samples):
    # print(real_samples.size())
    Tensor = torch.cuda.FloatTensor
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    if real_samples.ndim == 4:
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    else:
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
