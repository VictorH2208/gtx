import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, in_g, in_s, inter_channels):
        super(AttentionGate, self).__init__()
        self.Wg = nn.Conv2d(in_g, inter_channels, kernel_size=1, stride=2)
        self.Ws = nn.Conv2d(in_s, inter_channels, kernel_size=1)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = F.relu(Wg + Ws)
        out = self.psi(out)
        out = torch.sigmoid(out)
        out = F.interpolate(out, size=g.shape[2:], mode='bilinear', align_corners=False)
        return out * g
    
class UnetModel(nn.Module):
    def __init__(self, params):
        super(UnetModel, self).__init__()
        self.params = params

        nf2d = params['nFilters2D'] // 2
        nf3d = params['nFilters3D'] // 2

        # Optical branch (2D)
        self.op_branch = nn.Sequential(
            nn.Conv2d(2, nf2d, kernel_size=params['kernelConv2D'], stride=params['strideConv2D'], padding='same'),
            nn.ReLU(),
            nn.Dropout(0.75),
            nn.Conv2d(nf2d, nf2d, kernel_size=params['kernelConv2D'], stride=params['strideConv2D'], padding='same'),
            nn.ReLU(),
            nn.Dropout(0.75),
            nn.Conv2d(nf2d, nf2d, kernel_size=params['kernelConv2D'], stride=params['strideConv2D'], padding='same'),
            nn.ReLU(),
            nn.Dropout(0.75)
        )

        # Fluorescence branch (3D)
        self.fl_conv3d = nn.Sequential(
            nn.Conv3d(1, nf3d, kernel_size=params['kernelConv3D'], stride=params['strideConv3D'], padding='same'),
            nn.ReLU(),
            nn.Dropout(0.75),
            nn.Conv3d(nf3d, nf3d, kernel_size=params['kernelConv3D'], stride=params['strideConv3D'], padding='same'),
            nn.ReLU(),
            nn.Dropout(0.75),
            nn.Conv3d(nf3d, nf3d, kernel_size=params['kernelConv3D'], stride=params['strideConv3D'], padding='same'),
            nn.ReLU(),
            nn.Dropout(0.75)
        )

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Post-concat
        self.conv_post_1 = nn.Sequential(
            nn.Conv2d(nf2d + nf3d * params['nF'], 256, kernel_size=params['kernelConv2D'], stride=params['strideConv2D'], padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=params['kernelConv2D'], stride=params['strideConv2D'], padding='same'),
            nn.ReLU()
        )
        self.conv_post_2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=params['kernelConv2D'], stride=params['strideConv2D'], padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=params['kernelConv2D'], stride=params['strideConv2D'], padding='same'),
            nn.ReLU()
        )
        self.conv_post_3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=params['kernelConv2D'], stride=params['strideConv2D'], padding='same'),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=params['kernelConv2D'], stride=params['strideConv2D'], padding='same'),
            nn.ReLU()
        )

        # Attention Gates
        self.att1 = AttentionGate(1024, 512, 512)
        self.att2 = AttentionGate(512, 256, 256)
        self.att3 = AttentionGate(256, nf2d + nf3d * params['nF'], 128)

        # Decoder
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1024, 512, kernel_size=2, padding=1),
            nn.ReLU()
        )
        self.conv_dec1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=params['kernelConv2D'], padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=params['kernelConv2D'], padding='same'),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=2, padding=1),
            nn.ReLU()
        )
        self.conv_dec2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=params['kernelConv2D'], padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=params['kernelConv2D'], padding='same'),
            nn.ReLU()
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=2, padding=1),
            nn.ReLU()
        )
        self.conv_dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=params['kernelConv2D'], padding='same'),
            nn.ReLU()
        )

        # Output layers for QF and DF
        self.out_qf = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=params['kernelConv2D'], padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=params['kernelConv2D'], padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=params['kernelConv2D'], padding='same')
        )

        self.out_df = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=params['kernelConv2D'], padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=params['kernelConv2D'], padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=params['kernelConv2D'], padding='same')
        )

    def forward(self, inOP, inFL):
        op = self.op_branch(inOP)
        fl = self.fl_conv3d(inFL)
        fl = fl.view(fl.size(0), -1, fl.size(3), fl.size(4))  # Flatten channels
        x0 = torch.cat([op, fl], dim=1)

        x1 = self.conv_post_1(x0)
        x2 = self.conv_post_2(self.pool(x1))
        x3 = self.conv_post_3(self.pool(x2))

        att1 = self.att1(x3, x2)
        x = self.up1(x3)
        x = torch.cat([x, att1], dim=1)
        x = self.conv_dec1(x)

        att2 = self.att2(x, x1)
        x = self.up2(x)
        x = torch.cat([x, att2], dim=1)
        x = self.conv_dec2(x)

        att3 = self.att3(x, x0)
        x = self.up3(x)
        x = torch.cat([x, att3], dim=1)
        x = self.conv_dec3(x)

        qf = self.out_qf(x)
        df = self.out_df(x)
        return qf, df
