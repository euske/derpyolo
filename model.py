#!/usr/bin/env python
##
##  model.py - Mini YOLO model.
##
import torch
import torch.nn as nn


##  YOLONet
##
class YOLONet(nn.Module):

    IMAGE_SIZE = (224,224)
    GRID_SIZE = (7,7)

    def __init__(self, device, nvals):
        nn.Module.__init__(self)

        self.device = device
        self.nvals = nvals

        # [batchsize, 3, 224, 224]
        self.t1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
        ).to(device)
        # [batchsize, 32, 112, 112]
        self.t2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        ).to(device)
        # [batchsize, 64, 56, 56]
        self.t3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        ).to(device)
        # [batchsize, 128, 28, 28]
        self.t4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        ).to(device)
        # [batchsize, 256, 14, 14]
        self.t5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        ).to(device)
        # [batchsize, 512, 7, 7]
        self.t6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1,1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1,1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
        ).to(device)

        # [batchsize, 1024, 7, 7]
        self.t7 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=self.nvals, kernel_size=(1,1)),
            nn.Sigmoid(),
        ).to(device)
        # [batchsize, nvals, 7, 7]
        return

    def forward(self, x):
        x = x.to(self.device)
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)
        x = self.t5(x)
        x = self.t6(x)
        x = self.t7(x)
        # [batchsize, nvals, 7, 7]
        x = x.permute(0,2,3,1)
        # [batchsize, 7, 7, nvals]
        return x.cpu()

if __name__ == '__main__':
    net = YOLONet(0, 1)
    print(list(net.parameters()))
