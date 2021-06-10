import torch.nn as nn
import torch
from ultralight_conv import ultra_conv, ultra_one
from torch.nn.functional import interpolate


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ultra_conv(in_ch, out_ch, 3, padding=1,delta=0.25),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ultra_conv(out_ch, out_ch, 3, padding=1,delta=0.25),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class One_DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(One_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ultra_one(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ultra_one(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class SGU_Net(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SGU_Net, self).__init__()

        self.conv1 = One_DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.1)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(0.1)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(0.2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout2d(0.2)
        self.conv5 = DoubleConv(512, 1024)
        self.drop5 = nn.Dropout2d(0.3)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1536, 512)
        self.drop6 = nn.Dropout2d(0.2)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(768, 256)
        self.drop7 = nn.Dropout2d(0.2)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(384, 128)
        self.drop8 = nn.Dropout2d(0.1)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(192, 64)
        self.drop9 = nn.Dropout2d(0.1)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        d1 = self.drop1(p1)
        c2 = self.conv2(d1)
        p2 = self.pool2(c2)
        d2 = self.drop2(p2)
        c3 = self.conv3(d2)
        p3 = self.pool3(c3)
        d3 = self.drop3(p3)
        c4 = self.conv4(d3)
        p4 = self.pool4(c4)
        d4 = self.drop4(p4)
        c5 = self.conv5(d4)
        d5 = self.drop5(c5)
        up_6 = interpolate(d5,scale_factor=2,mode="bilinear")
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        d6 = self.drop6(c6)
        up_7 = interpolate(d6, scale_factor=2, mode="bilinear")
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        d7 = self.drop7(c7)
        up_8 = interpolate(d7, scale_factor=2, mode="bilinear")
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        d8 = self.drop8(c8)
        up_9 = interpolate(d8, scale_factor=2, mode="bilinear")
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        d9 = self.drop9(c9)
        c10 = self.conv10(d9)
        out = nn.Sigmoid()(c10)
        return out








