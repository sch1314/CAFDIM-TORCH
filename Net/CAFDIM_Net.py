
import torch.nn as nn
import torch
import torch.nn.functional as F
from CNN_Block import CNN_Unet
from EagleUnet import EagleUNet


class ConvBnRelu2d(nn.Module):

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


class BasicBlockTheta(nn.Module):

    def __init__(self, features=32):
        super(BasicBlockTheta, self).__init__()


        self.conv_D = nn.Conv2d(1, features, (3, 3), stride=1, padding=1)

        self.conv1_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv2_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv3_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv4_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)


        self.conv1_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv2_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv3_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv4_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv_G = nn.Conv2d(features, 1, (3, 3), stride=1, padding=1)  # 输出

    def forward(self, x, soft_thr):

        x_input = x
        x_D = self.conv_D(x)
        x = self.conv1_forward(x_D)
        x = F.relu(x)
        x = self.conv2_forward(x)
        x = F.relu(x)
        x = self.conv3_forward(x)
        x = F.relu(x)
        x_forward = self.conv4_forward(x)

        x_st = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - soft_thr))
        x = self.conv1_backward(x_st)
        x = F.relu(x)
        x = self.conv2_backward(x)
        x = F.relu(x)
        x = self.conv3_backward(x)
        x = F.relu(x)
        x_backward = self.conv4_backward(x)
        x_G = self.conv_G(x_backward)
        x_pred = F.relu(x_input + x_G)

        return x_pred


class CNNBlock2D(nn.Module):

    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(CNNBlock2D, self).__init__()
        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
        )

        self.conv_x = None

        if in_chl != out_chl:
            self.conv_x = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False,
                                       is_relu=False)

    def forward(self, x):

        conv_out = self.encode(x)

        if self.conv_x is None:
            res_out = F.leaky_relu(conv_out + x)

        else:

            res_out = F.leaky_relu(conv_out + self.conv_x(x))

        return res_out


class EncodeResBlock2D(nn.Module):

    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(EncodeResBlock2D, self).__init__()

        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
        )

        self.conv_x = None
        if in_chl != out_chl:
            self.conv_x = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False,
                                       is_relu=False)


    def forward(self, x):

        conv_out = self.encode(x)

        if self.conv_x is None:

            res_out = F.leaky_relu(conv_out + x)

        else:

            res_out = F.leaky_relu(conv_out + self.conv_x(x))

        down_out = F.max_pool2d(res_out, kernel_size=2, stride=2, padding=0, ceil_mode=True)


        return res_out, down_out


class DecodeResBlock2D(nn.Module):

    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(DecodeResBlock2D, self).__init__()

        self.encode = nn.Sequential(

            ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
        )

        self.conv_x = None


        if in_chl != out_chl:
            self.conv_x = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False,
                                       is_relu=False)

    def forward(self, x, skip_x):

        _, _, H, W = skip_x.size()

        up_out = F.upsample(x, size=(H, W), mode='bilinear')

        conv_out = self.encode(torch.cat([up_out, skip_x], 1))

        if self.conv_x is None:
            res_out = F.leaky_relu(conv_out + up_out)

        else:

            res_out = F.leaky_relu(conv_out + self.conv_x(up_out))

        return res_out


from Net.CAFB_Block import CAFB
class PR_Net(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, kernel_size=3, model_chl=32):
        super(PR_Net, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.conv_start = ConvBnRelu2d(in_chl, model_chl, kernel_size=kernel_size, dilation=1, stride=1,
                                       groups=1)

        self.encoder1 = EncodeResBlock2D(model_chl, model_chl)
        self.encoder2 = EncodeResBlock2D(model_chl, model_chl * 2)
        self.encoder3 = EncodeResBlock2D(model_chl * 2, model_chl * 4)  # 64--》128

        self.conv_center =CAFB(model_chl * 4, model_chl * 8)

        self.decoder3 = DecodeResBlock2D(model_chl * 8, model_chl * 4)
        self.decoder2 = DecodeResBlock2D(model_chl * 4, model_chl * 2)  # 64+64-->32+32
        self.decoder1 = DecodeResBlock2D(model_chl * 2, model_chl)  # 32+32-->32

        self.conv_end = ConvBnRelu2d(model_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1,
                                     is_relu=False)

    def forward(self, x):

        x_start = self.conv_start(x)

        res_x1, down_x1 = self.encoder1(x_start)
        res_x2, down_x2 = self.encoder2(down_x1)
        res_x3, down_x3 = self.encoder3(down_x2)

        x_center = self.conv_center(down_x3)


        out_x3 = self.decoder3(x_center, res_x3)
        out_x2 = self.decoder2(out_x3, res_x2)
        out_x1 = self.decoder1(out_x2, res_x1)

        out = F.leaky_relu(self.conv_end(out_x1) + torch.unsqueeze(x[:, -1, :, :], 1))
        return out
class RR_Net(nn.Module):

    def __init__(self, in_chl=1, out_chl=1, kernel_size=3, model_chl=32):
        super(RR_Net, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl
        self.conv_start = ConvBnRelu2d(in_chl, model_chl, kernel_size=kernel_size, dilation=1, stride=1,
                                       groups=1)
        self.encoder1 = EncodeResBlock2D(model_chl, model_chl)
        self.encoder2 = EncodeResBlock2D(model_chl, model_chl * 2)  # 32-->64


        self.conv_center = CAFB(model_chl * 2, model_chl * 4)
        self.decoder2 = DecodeResBlock2D(model_chl * 4, model_chl * 2)
        self.decoder1 = DecodeResBlock2D(model_chl * 2, model_chl)  # 32+32(64)-->32

        self.conv_end = ConvBnRelu2d(model_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1,
                                     is_relu=False)

    def forward(self, x):
        x_start = self.conv_start(x)

        res_x1, down_x1 = self.encoder1(x_start)
        res_x2, down_x2 = self.encoder2(down_x1)

        x_center = self.conv_center(down_x2)

        out_x2 = self.decoder2(x_center, res_x2)
        out_x1 = self.decoder1(out_x2, res_x1)
        out = F.leaky_relu(self.conv_end(out_x1) + torch.unsqueeze(x[:, -1, :, :], 1))  # step5 32-->1

        return out


class BasicBlock(nn.Module):

    def __init__(self, recon_op):
        super(BasicBlock, self).__init__()
        self.Sp = nn.Softplus()
        self.recon_op = recon_op

    def forward(self, x, b, lambda_step):

        rk = x - lambda_step * self.recon_op.backprojection(
            self.recon_op.filter_sinogram(
                self.recon_op.forward(x / 1024) - b)) * 1024

        return rk

class InitEnhance_Net3(nn.Module):  # DREAM_UNetL3


    def __init__(self, in_chl=1, out_chl=1, kernel_size=3, model_chl=32):
        super(InitEnhance_Net3, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.conv_start = ConvBnRelu2d(in_chl, model_chl, kernel_size=kernel_size, dilation=1, stride=1,
                                       groups=1)

        self.encoder1 = EncodeResBlock2D(model_chl, model_chl)

        self.encoder2 = EncodeResBlock2D(model_chl, model_chl * 2)
        self.encoder3 = EncodeResBlock2D(model_chl * 2, model_chl * 4)

        self.conv_center = CNNBlock2D(model_chl * 4, model_chl * 8)
        self.decoder3 = DecodeResBlock2D(model_chl * 8, model_chl * 4)
        self.decoder2 = DecodeResBlock2D(model_chl * 4, model_chl * 2)
        self.decoder1 = DecodeResBlock2D(model_chl * 2, model_chl)

        self.conv_end = ConvBnRelu2d(model_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1,
                                     is_relu=False)

    def forward(self, x):

        x_start = self.conv_start(x)

        res_x1, down_x1 = self.encoder1(x_start)
        res_x2, down_x2 = self.encoder2(down_x1)
        res_x3, down_x3 = self.encoder3(down_x2)

        x_center = self.conv_center(down_x3)

        out_x3 = self.decoder3(x_center, res_x3)
        out_x2 = self.decoder2(out_x3, res_x2)
        out_x1 = self.decoder1(out_x2, res_x1)


        out = F.leaky_relu(self.conv_end(out_x1) + torch.unsqueeze(x[:, -1, :, :], 1))
        return out
class CAFDIM(nn.Module):

    def __init__(self, recon_op, iter_block=3, net_chl=32):
        super(CAFDIM, self).__init__()
        self.iter_block = iter_block
        self.net_chl = net_chl
        self.recon_op = recon_op

        self.net = nn.ModuleList()

        self.InitEnhance = InitEnhance_Net3(in_chl=1, out_chl=1, kernel_size=3, model_chl=self.net_chl)
        self.PRNet = PR_Net(in_chl=1, out_chl=1, kernel_size=3, model_chl=self.net_chl)
        for i in range(self.iter_block):
            self.net.append(RR_Net(in_chl=i + 2, out_chl=1, kernel_size=3, model_chl=self.net_chl))
        self.bb = BasicBlock(recon_op)
        self.LayerNo = iter_block
        self.mu = torch.nn.Parameter(torch.FloatTensor(self.LayerNo).fill_(0.001))

        self.encoder0 = BasicBlockTheta(features=32)
        self.theta = torch.nn.Parameter(torch.FloatTensor(self.iter_block).fill_(20))
        self.rho = torch.nn.Parameter(torch.FloatTensor(self.LayerNo).fill_(0.001))
        self.EagleUUnet = EagleUNet()
        self.Unet = CNN_Unet(in_chans=1, out_chans=1)


    def forward(self, proj, ldct, mask):

        img_current = self.InitEnhance(ldct)
        img_dense = img_current
        y_add = img_current
        y_old = img_current

        for i in range(self.iter_block):

            img_current = self.bb(y_add, proj, F.relu(self.mu[i])) + self.Unet(y_add)


            proj_current = self.recon_op.forward(img_current / 1024)
            proj_current = self.encoder0(proj_current, F.relu(self.theta[i]))
            proj_net = self.PRNet(proj_current)
            proj_wrt = proj_net * (1 - mask) + proj * mask
            img_error = self.recon_op.backprojection(self.recon_op.filter_sinogram(proj_current - proj_wrt)) * 1024
            img_error_net = self.EagleUUnet(img_error)
            img_current = self.net[i](torch.cat([img_dense, img_current + img_error_net], 1))
            y_add = img_current + F.relu(self.rho[i]) * (img_current - y_old)
            y_old = img_current

            img_dense = torch.cat([img_dense, img_current], 1)

        return proj_net, img_current

