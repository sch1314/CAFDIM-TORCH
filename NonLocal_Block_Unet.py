import torch.nn as nn
import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
import numpy as np


def create_mask(neighborhood_size, SIZE):
    mask = torch.zeros([SIZE, SIZE], dtype=torch.bool)
    for j in range(0, SIZE):
        for i in range(0, int(neighborhood_size * np.sqrt(SIZE)), int(np.sqrt(SIZE))):
            for k in range(neighborhood_size):
                mask[j, i + k] = 1
    return mask


"""
NonLocal放到了Unet里面
NonLocal 类是一个非局部自注意力层，它可以增强模型对于图像中远距离依赖的感知能力。

"""


class NonLocal(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation, SIZE):
        super(NonLocal, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1))
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #
        self.SIZE = int(SIZE)
        self.neighborhood_size = 3
        # self.n_neighbour=9
        self.mask = create_mask(self.neighborhood_size, self.SIZE)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
            inputs : B：batch_size,C：chanl,W*H

                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)

                                    输入：
                                        x：输入特征图（B x C x W x H）
                                    退货：
                                    out：自我关注值+输入特性
                                    注意：B X N X N（N为宽度*高度）
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        #
        mask = self.mask.repeat(m_batchsize, 1, 1)

        energy[~mask] = 0

        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        return torch.cat((out, x), dim=1)


class UNet(nn.Module):

    def __init__(self, image_size):
        super(UNet, self).__init__()
        self.input_channel = 1
        self.inter_channel = 64
        self.conv1 = nn.Sequential(nn.Conv2d(1, self.inter_channel, 5, padding=2),
                                   nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(nn.Conv2d(self.inter_channel, self.inter_channel, 5, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.inter_channel, self.inter_channel, 5, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.inter_channel, self.inter_channel, 5, padding=2),
                                    nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.layer2 = nn.Sequential(nn.Conv2d(self.inter_channel, self.inter_channel, 5, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.inter_channel, self.inter_channel, 5, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.inter_channel, self.inter_channel, 5, padding=2),
                                    nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.layer3 = nn.Sequential(NonLocal(64, 'relu', (image_size/8) * (image_size/8)),
                                    nn.Conv2d(2 * self.inter_channel, self.inter_channel, 5, padding=2),
                                    nn.ReLU(inplace=True))
        #                             # # NonLocal(1, 'relu', image_size / 4 * (image_size / 4)),
        #                             # NonLocal(64, 'relu', (image_size/8) * (image_size/8)),
        #                             # nn.Conv2d(2 * self.inter_channel, self.inter_channel, 5, padding=2),
        #                             # nn.ReLU(inplace=True),
        #                             # NonLocal(64, 'relu', (image_size/8) * (image_size/8)),
        #                             # nn.Conv2d(2 * self.inter_channel, self.inter_channel, 5, padding=2),
        #                             # nn.ReLU(inplace=True),
        #                             # NonLocal(64, 'relu', (image_size/8) * (image_size/8)),
        #                             # nn.Conv2d(2 * self.inter_channel, self.inter_channel, 5, padding=2),
        #                             # nn.ReLU(inplace=True))
        # # self.layer3 = nn.Sequential(NonLocal(64, 'relu', (image_size / 8) * (image_size / 8)),
        # #                             nn.Conv2d(2 * self.inter_channel, self.inter_channel, 5, padding=2),
        # #                             nn.ReLU(inplace=True))
        self.pool3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer4 = nn.Sequential(nn.Conv2d(2 * self.inter_channel, self.inter_channel, 5, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.inter_channel, self.inter_channel, 5, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.inter_channel, self.inter_channel, 5, padding=2),
                                    nn.ReLU(inplace=True))
        self.pool4 = nn.Upsample(scale_factor=2, mode='nearest')

        self.layer5 = nn.Sequential(nn.Conv2d(2 * self.inter_channel, self.inter_channel, 5, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.inter_channel, self.inter_channel, 5, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.inter_channel, self.inter_channel, 5, padding=2),
                                    nn.ReLU(inplace=True))
        self.conv2 = nn.Conv2d(self.inter_channel, 1, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)

        x1 = self.layer1(x)

        x = self.pool1(x1)

        x2 = self.layer2(x)

        x = self.pool2(x2)

        # x = self.layer3(x)

        x = self.pool3(x)

        x = self.layer4(torch.cat((x2, x), 1))

        x = self.pool4(x)

        x = self.layer5(torch.cat((x1, x), 1))

        x = self.conv2(x)
        return x