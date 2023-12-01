# from: https://github.com/Skybird1101/EBUNet
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["EBUNet"]


def Split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2


class conv3x3_resume(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.conv3x3 = Conv(nIn // 2, nIn // 2, kSize, 1, padding=1, bn_acti=True)
        self.conv1x1_resume = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.conv3x3(input)
        output = self.conv1x1_resume(output)
        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output


class EBU(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)

        self.dconv_left = Conv(nIn // 4, nIn // 4, (dkSize, dkSize), 1,
                               padding=(1, 1), groups=nIn // 4, bn_acti=True)

        self.dconv_right = Conv(nIn // 4, nIn // 4, (dkSize, dkSize), 1,
                                padding=(1 * d, 1 * d), groups=nIn // 4, dilation=(d, d), bn_acti=True)
        self.bn_relu_1 = BNPReLU(nIn)

        self.conv3x3_resume = conv3x3_resume(nIn, nIn, (dkSize, dkSize), 1,
                                             padding=(1, 1), bn_acti=True)

    def forward(self, input):
        output = self.conv3x3(input)
        x1, x2 = Split(output)
        left = self.dconv_left(x1)

        right = self.dconv_right(x2)

        left = left + right
        right = right + left
        output = torch.cat((left, right), 1)
        output = self.conv3x3_resume(output)

        return self.bn_relu_1(output + input)


class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)

        return input


class SpatialAttention(nn.Module):
    def __init__(self, k=3):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=k, padding=k // 2,
                              bias=False)  # infer a one-channel attention map

    def forward(self, ftr):
        # ftr: [B, C, H, W]
        ftr_avg = torch.mean(ftr, dim=1, keepdim=True)  # [B, 1, H, W], average
        ftr_max, _ = torch.max(ftr, dim=1, keepdim=True)  # [B, 1, H, W], max
        ftr_cat = torch.cat([ftr_avg, ftr_max], dim=1)  # [B, 2, H, W]
        att_map = F.sigmoid(self.conv(ftr_cat))  # [B, 1, H, W]
        return att_map


class CAM(nn.Module):
    def __init__(self, channel, k_size=3):
        super(CAM, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        y = self.conv(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class LAD(nn.Module):
    def __init__(self, c1=16, c2=32, classes=19):
        super(LAD, self).__init__()
        self.c1, self.c2 = c1, c2
        self.LMFFNet_Block_2 = nn.Sequential()

        self.mid_layer_1x1 = Conv(128 + 3, c1, 1, 1, padding=0, bn_acti=False)

        self.deep_layer_1x1 = Conv(256 + 3, c2, 1, 1, padding=0, bn_acti=False)

        self.DwConv1 = Conv(self.c1 + self.c2, self.c1 + self.c2, (3, 3), 1, padding=(1, 1),
                            groups=self.c1 + self.c2, bn_acti=True)

        self.PwConv1 = Conv(self.c1 + self.c2, classes, 1, 1, padding=0, bn_acti=False)
        self.sam = SpatialAttention()
        self.cam = ChannelAttention(32)

    def forward(self, x1, x2):
        x2_size = x2.size()[2:]
        # x1: 1/4 feature map, low-level feature map
        # x2:1/8 feature map,high-level feature map
        x1_ = self.mid_layer_1x1(x1)
        spatial_weights = self.sam(x1_)
        x1_ = spatial_weights * x1_
        x2_ = self.deep_layer_1x1(x2)
        channel_weights = self.cam(x2_)
        x2_ = x2_ * channel_weights
        x2_ = F.interpolate(x2_, [x2_size[0] * 2, x2_size[1] * 2], mode='bilinear', align_corners=False)

        x1_x2_cat = torch.cat([x1_, x2_], 1)

        x1_x2_cat = self.DwConv1(x1_x2_cat)
        x1_x2_cat = self.PwConv1(x1_x2_cat)

        o = F.interpolate(x1_x2_cat, [x2_size[0] * 8, x2_size[1] * 8], mode='bilinear', align_corners=False)

        return o


class IPAM(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(IPAM, self).__init__()

        self.partition_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.conv2x2 = Conv(ch_in, ch_in, 2, 1, padding=(0, 0), groups=ch_in, bn_acti=False)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cam = CAM(ch_in)

    def forward(self, x):
        o1 = self.partition_pool(x)

        o1 = self.conv2x2(o1)

        o2 = self.global_pool(x)

        o_sum = o1 + o2
        w = self.cam(o_sum)
        o = w * x

        return o


class FFM_B(nn.Module):
    def __init__(self, ch_in, ch_pmca):
        super(FFM_B, self).__init__()
        self.PMCA = IPAM(ch_in=ch_pmca, reduction=8)
        self.bn_prelu = BNPReLU(ch_in)
        self.conv1x1 = Conv(ch_in, ch_in, 1, 1, padding=0, bn_acti=False)

    def forward(self, x):
        x1, x2, x3 = x
        x2 = self.PMCA(x2)
        o = self.bn_prelu(torch.cat([x1, x2, x3], 1))
        o = self.conv1x1(o)
        return o


class EBUNet(nn.Module):
    def __init__(self, classes=19, block_1=3, block_2=10):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )

        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        self.down_2 = InputInjection(2)  # down-sample the image 2 times
        self.down_3 = InputInjection(3)  # down-sample the image 3 times

        self.bn_prelu_1 = BNPReLU(32 + 3)

        # EBU Block 1
        self.downsample_1 = DownSamplingBlock(32 + 3, 64)
        self.EBU_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.EBU_Block_1.add_module("EBU_Module_1_" + str(i), EBU(64, d=2))
        self.bn_prelu_2 = BNPReLU(128 + 3)
        self.ffmb_1 = FFM_B(ch_in=128 + 3, ch_pmca=64)

        # EBU Block 2

        dilation_block_2 = [2, 2, 4, 4, 6, 6, 8, 8, 16, 16]

        self.downsample_2 = DownSamplingBlock(128 + 3, 128)
        self.EBU_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.EBU_Block_2.add_module("EBU_Module_2_" + str(i),
                                        EBU(128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(256 + 3)
        self.ffmb_2 = FFM_B(ch_in=259, ch_pmca=128)
        self.lad = LAD(classes=classes)
        self.classifier = nn.Sequential(Conv(259, classes, 1, 1, padding=0))

    def forward(self, input):

        output0 = self.init_conv(input)

        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)
        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))

        # EBU Block 1
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.EBU_Block_1(output1_0)
        temp = output1, output1_0, down_2
        output1_cat = self.ffmb_1(temp)

        # EBU Block 2
        output2_0 = self.downsample_2(output1_cat)
        output2 = self.EBU_Block_2(output2_0)
        temp = output2, output2_0, down_3
        output2_cat = self.ffmb_2(temp)

        out = self.lad(output1_cat, output2_cat)

        return out


if __name__ == '__main__':
    from torchstat import stat

    model = EBUNet()
    stat(model, (3, 720, 960))
