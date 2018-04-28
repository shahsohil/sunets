import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torchvision import models
import os
from itertools import chain
from ptsemseg.loss import cross_entropy2d, prediction_stat, prediction_stat_confusion_matrix

checkpoint = 'pretrained/ResNet'
res18_path = os.path.join(checkpoint, 'resnet18-5c106cde.pth')
res101_path = os.path.join(checkpoint, 'resnet101-5d3b4d8f.pth')

mom_bn = 0.05
dilation = {'16':1, '8':2}

class d_resnet18(nn.Module):
    def __init__(self, num_classes, pretrained=True, use_aux=True, ignore_index=-1, output_stride='16'):
        super(d_resnet18, self).__init__()
        self.use_aux = use_aux
        self.num_classes = num_classes
        resnet = models.resnet18()
        if pretrained:
            resnet.load_state_dict(torch.load(res18_path))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        d = dilation[output_stride]
        if d > 1:
            for n, m in self.layer3.named_modules():
                if '0.conv1' in n:
                    m.dilation, m.padding, m.stride = (1, 1), (1, 1), (1, 1)
                elif 'conv1' in n:
                    m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if '0.conv1' in n:
                m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
            elif 'conv1' in n:
                m.dilation, m.padding, m.stride = (2*d, 2*d), (2*d, 2*d), (1, 1)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (2*d, 2*d), (2*d, 2*d), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        for n, m in chain(self.layer0.named_modules(), self.layer1.named_modules(), self.layer2.named_modules(), self.layer3.named_modules(), self.layer4.named_modules()):
            if 'downsample.1' in n:
                m.momentum = mom_bn
            elif 'bn' in n:
                m.momentum = mom_bn

        self.final = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=mom_bn),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        self.mceloss = cross_entropy2d(ignore=ignore_index, size_average=False)

    def forward(self, x, labels, th=1.0):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final(x)
        x = F.upsample(x, x_size[2:], mode='bilinear')

        if labels is not None:
            losses, total_valid_pixel = self.mceloss(x, labels, th=th)

            classwise_pixel_acc, classwise_gtpixels, classwise_predpixels = prediction_stat([x], labels, self.num_classes)

            # Need to perform this operation for MultiGPU
            classwise_pixel_acc = Variable(torch.FloatTensor([classwise_pixel_acc]).cuda())
            classwise_gtpixels = Variable(torch.FloatTensor([classwise_gtpixels]).cuda())
            classwise_predpixels = Variable(torch.FloatTensor([classwise_predpixels]).cuda())

            return x, losses, classwise_pixel_acc, classwise_gtpixels, classwise_predpixels, total_valid_pixel
        else:
            return x


class d_resnet101(nn.Module):
    def __init__(self, num_classes, pretrained=True, use_aux=True, ignore_index=-1, output_stride='16'):
        super(d_resnet101, self).__init__()
        self.use_aux = use_aux
        self.num_classes = num_classes
        resnet = models.resnet101()
        if pretrained:
            resnet.load_state_dict(torch.load(res101_path))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        d = dilation[output_stride]
        if d > 1:
            for n, m in self.layer3.named_modules():
                if '0.conv2' in n:
                    m.dilation, m.padding, m.stride = (1, 1), (1, 1), (1, 1)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if '0.conv2' in n:
                m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (2*d, 2*d), (2*d, 2*d), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        for n, m in chain(self.layer0.named_modules(), self.layer1.named_modules(), self.layer2.named_modules(), self.layer3.named_modules(), self.layer4.named_modules()):
            if 'downsample.1' in n:
                m.momentum = mom_bn
            elif 'bn' in n:
                m.momentum = mom_bn

        self.final = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=mom_bn),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        self.mceloss = cross_entropy2d(ignore=ignore_index)

    def forward(self, x, labels, th=1.0):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final(x)
        x = F.upsample(x, x_size[2:], mode='bilinear')

        if labels is not None:
            losses, total_valid_pixel = self.mceloss(x, labels, th=th)

            classwise_pixel_acc, classwise_gtpixels, classwise_predpixels = prediction_stat([x], labels, self.num_classes)

            # Need to perform this operation for MultiGPU
            classwise_pixel_acc = Variable(torch.FloatTensor([classwise_pixel_acc]).cuda())
            classwise_gtpixels = Variable(torch.FloatTensor([classwise_gtpixels]).cuda())
            classwise_predpixels = Variable(torch.FloatTensor([classwise_predpixels]).cuda())

            return x, losses, classwise_pixel_acc, classwise_gtpixels, classwise_predpixels, total_valid_pixel
        else:
            return x