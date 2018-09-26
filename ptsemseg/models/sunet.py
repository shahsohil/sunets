import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from ..loss import cross_entropy2d, prediction_stat, prediction_stat_confusion_matrix

from torch.autograd import Variable

checkpoint = 'pretrained/SUNets'
sunet64_path = os.path.join(os.path.dirname(__file__), checkpoint, 'checkpoint_64_2441_residual.pth.tar')
sunet128_path = os.path.join(os.path.dirname(__file__), checkpoint, 'checkpoint_128_2441_residual.pth.tar')
sunet7128_path = os.path.join(os.path.dirname(__file__), checkpoint, 'checkpoint_128_2771_residual.pth.tar')

mom_bn = 0.01
output_stride = {'32':3, '16':2, '8':1}

class UNetConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, dprob, mod_in_planes, is_input_bn, dilation):
        super(UNetConv, self).__init__()
        if mod_in_planes:
            if is_input_bn:
                self.add_module('bn0', nn.BatchNorm2d(in_planes))
                self.add_module('relu0', nn.ReLU(inplace=True))
            self.add_module('conv0', nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=True))
            self.add_module('dropout0', nn.Dropout(p=dprob))
            in_planes = out_planes

        self.add_module('bn1', nn.BatchNorm2d(in_planes))
        self.add_module('relu1', nn.ReLU(inplace=True))
        if dilation > 1:
            self.add_module('conv1', nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation, stride=1, dilation=dilation, bias=True))
        else:
            self.add_module('conv1', nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=2, bias=True))
        self.add_module('dropout1', nn.Dropout(p=dprob))
        self.add_module('bn2', nn.BatchNorm2d(out_planes))
        self.add_module('relu2', nn.ReLU(inplace=True))
        if dilation > 1:
            self.add_module('conv2', nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=2*dilation, stride=1, dilation=2*dilation, bias=True))
        else:
            self.add_module('conv2', nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, stride=1, bias=True))
        self.add_module('dropout2', nn.Dropout(p=dprob))


class UNetDeConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, dprob, mod_in_planes, max_planes, dilation, output_padding=1):
        super(UNetDeConv, self).__init__()
        self.add_module('bn0', nn.BatchNorm2d(in_planes))
        self.add_module('relu0', nn.ReLU(inplace=True))
        if dilation > 1:
            self.add_module('deconv0',nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=1, padding=2 * dilation,
                                                         dilation=2 * dilation, bias=True))
        else:
            self.add_module('deconv0',nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1,
                                                         output_padding=output_padding, bias=True))
        self.add_module('dropout0', nn.Dropout(p=dprob))
        self.add_module('bn1', nn.BatchNorm2d(out_planes))
        self.add_module('relu1', nn.ReLU(inplace=True))
        if dilation > 1:
            self.add_module('deconv1', nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=dilation, stride=1, dilation=dilation, bias=True))
        else:
            self.add_module('deconv1', nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, stride=1, bias=True))
        self.add_module('dropout1', nn.Dropout(p=dprob))
        if mod_in_planes:
            self.add_module('bn2', nn.BatchNorm2d(out_planes))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('conv2', nn.Conv2d(out_planes, max_planes, kernel_size=1, bias=True))
            self.add_module('dropout2', nn.Dropout(p=dprob))


class UNetModule(nn.Module):
    def __init__(self, in_planes, nblock, filter_size, dprob, in_dim, index, max_planes, atrous=0):
        super(UNetModule, self).__init__()
        self.nblock = nblock
        self.in_dim = np.array(in_dim, dtype=float)
        self.down = nn.ModuleList([])
        self.up = nn.ModuleList([])
        self.upsample = None
        if in_planes != max_planes:
            self.bn = nn.Sequential(OrderedDict([
                ('bn0', nn.BatchNorm2d(in_planes)),
                ('relu0', nn.ReLU(inplace=True))
            ]))
            self.upsample = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(in_planes, max_planes, kernel_size=1, stride=1, bias=True))
            ]))
        for i in range(nblock):
            if i == 0:
                in_ = in_planes
            else:
                in_ = filter_size
            self.down.append(UNetConv(in_, filter_size, dprob, index and (i == 0), in_planes == max_planes, (2**i)*atrous))
            if i > 1:
                self.down[-1].conv.weight = self.conv_1.conv.weight
                self.down[-1].conv1.weight = self.conv_1.conv1.weight
            if i == nblock-1:
                out_ = filter_size
            else:
                out_ = 2 * filter_size
            self.up.append(UNetDeConv(out_, filter_size, dprob, index and (i == 0), max_planes, (2**i)*atrous, output_padding=1-int(np.mod(self.in_dim,2))))
            if i > 0 and i < nblock-1:
                self.up[-1].deconv.weight = self.deconv_0.deconv.weight
                self.up[-1].deconv1.weight = self.deconv_0.deconv1.weight
            self.in_dim = np.ceil(self.in_dim / 2)

    def forward(self, x):
        xs = []
        if self.upsample is not None:
            x = self.bn(x)
        xs.append(x)
        for i, down in enumerate(self.down):
            xout = down(xs[-1])
            xs.append(xout)

        out = xs[-1]
        for i, (x_skip, up) in reversed(list(enumerate(zip(xs[:-1], self.up)))):
            out = up(out)
            if i:
                out = torch.cat([out, x_skip], 1)
            else:
                if self.upsample is not None:
                    x_skip = self.upsample(x_skip)
                out += x_skip
        return out


class Transition(nn.Sequential):
    def __init__(self, in_planes, out_planes, dprob):
        super(Transition, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_planes))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=True))
        self.add_module('dropout', nn.Dropout(p=dprob))


class ResidualBlock(nn.Sequential):
    def __init__(self, in_planes, out_planes, dprob, stride=1):
        super(ResidualBlock, self).__init__()
        self.bn = nn.Sequential(nn.BatchNorm2d(in_planes),
                                 nn.ReLU(inplace=True))
        self.conv = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride, bias=True),
                                  nn.Dropout(p=dprob),
                                  nn.BatchNorm2d(out_planes),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, stride=1, bias=True),
                                  nn.Dropout(p=dprob))
        if stride > 1:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes,out_planes, kernel_size=1, stride=stride, bias=True))
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.bn(x)
        residual = out
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        out = self.conv(out)
        out += residual
        return out


class Stackedunet_imagenet(nn.Module):
    filter_factors = [1, 1, 1, 1]

    def __init__(self, in_dim=224, start_planes=16, filters_base=64, num_classes=1000, depth=1, dprob=1e-7, ost='32'):
        super(Stackedunet_imagenet, self).__init__()
        self.start_planes = start_planes
        self.depth = depth
        feature_map_sizes = [filters_base * s for s in self.filter_factors]

        if filters_base == 128 and depth == 4:
            output_features = [512, 1024, 1536, 2048]
        elif filters_base == 128 and depth == 7:
            output_features = [512, 1280, 2048, 2304]
        elif filters_base == 64 and depth == 4:
            output_features = [256, 512, 768, 1024]

        num_planes = start_planes
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, num_planes, kernel_size=7, stride=2, padding=3, bias=True)),
        ]))
        in_dim = in_dim // 2
        self.features.add_module('residual1', ResidualBlock(num_planes, 2*num_planes, dprob, stride=2))
        num_planes *= 2
        in_dim = in_dim // 2

        block_depth = (2, depth, depth, 1)
        nblocks = 2
        for j, d in enumerate(block_depth):
            if j == len(block_depth)-1:
                nblocks = 1
            for i in range(d):
                block = UNetModule(num_planes, nblocks, feature_map_sizes[j], dprob, in_dim, 1, output_features[j], (j-output_stride[ost])*2)
                self.features.add_module('unet%d_%d'% (j+1, i), block)
                num_planes = output_features[j]
            if j != len(block_depth)-1:
                if j > output_stride[ost]-1:
                    self.features.add_module('avgpool%d' % (j + 1), nn.AvgPool2d(kernel_size=1, stride=1))
                else:
                    self.features.add_module('avgpool%d'%(j+1),nn.AvgPool2d(kernel_size=2, stride=2))
                    in_dim = in_dim // 2

        self.features.add_module('bn2', nn.BatchNorm2d(num_planes))
        self.linear = nn.Linear(num_planes, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = F.relu(out, inplace=False)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def stackedunet7128(output_stride='32'):
    return Stackedunet_imagenet(in_dim=512, start_planes=64, filters_base=128, num_classes=1000, depth=7, ost=output_stride)


def stackedunet128(output_stride='32'):
    return Stackedunet_imagenet(in_dim=512, start_planes=64, filters_base=128, num_classes=1000, depth=4, ost=output_stride)


def stackedunet64(output_stride='32'):
    return Stackedunet_imagenet(in_dim=512, start_planes=64, filters_base=64, num_classes=1000, depth=4, ost=output_stride)

# Dilated SUNET models
class d_sunet64(nn.Module):
    def __init__(self, num_classes, pretrained=True, ignore_index=-1, weight=None, output_stride='16'):
        super(d_sunet64, self).__init__()
        self.num_classes = num_classes
        sunet = stackedunet64(output_stride=output_stride)
        sunet = torch.nn.DataParallel(sunet, device_ids=range(torch.cuda.device_count())).cuda()
        if pretrained:
            checkpoint = torch.load(sunet64_path)
            sunet.load_state_dict(checkpoint['state_dict'])

        self.features = sunet.module.features

        for n, m in self.features.named_modules():
            if 'bn' in n:
                m.momentum = mom_bn

        # hack needed as batchnorm modules were not named as bn in residual block
        for n, m in self.features.residual1.conv.named_modules():
            if '2' in n:
                m.momentum = mom_bn

        self.final = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1024, num_classes, kernel_size=1)),
        ]))

        self.mceloss = cross_entropy2d(ignore=ignore_index, size_average=False, weight=weight)

    def forward(self, x, labels=None, th=1.0):
        x_size = x.size()
        x = self.features(x)
        x = F.relu(x, inplace=False)
        x = self.final(x)
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False)

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

class d_sunet128(nn.Module):
    def __init__(self, num_classes, pretrained=True, ignore_index=-1, weight=None, output_stride='16'):
        super(d_sunet128, self).__init__()
        self.num_classes = num_classes
        sunet = stackedunet128(output_stride=output_stride)
        sunet = torch.nn.DataParallel(sunet, device_ids=range(torch.cuda.device_count())).cuda()
        if pretrained:
            checkpoint = torch.load(sunet128_path)
            sunet.load_state_dict(checkpoint['state_dict'])

        self.features = sunet.module.features

        for n, m in self.features.named_modules():
            if 'bn' in n:
                m.momentum = mom_bn

        # hack needed as batchnorm modules were not named as bn in residual block
        for n, m in self.features.residual1.conv.named_modules():
            if '2' in n:
                m.momentum = mom_bn

        self.final = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(2048, num_classes, kernel_size=1)),
        ]))

        self.mceloss = cross_entropy2d(ignore=ignore_index, size_average=False, weight=weight)

    def forward(self, x, labels=None, th=1.0):
        x_size = x.size()
        x = self.features(x)
        x = F.relu(x, inplace=False)
        x = self.final(x)
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False)

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

class d_sunet7128(nn.Module):
    def __init__(self, num_classes, pretrained=True, ignore_index=-1, weight=None, output_stride='16'):
        super(d_sunet7128, self).__init__()
        self.num_classes = num_classes
        sunet = stackedunet7128(output_stride=output_stride)
        sunet = torch.nn.DataParallel(sunet, device_ids=range(torch.cuda.device_count())).cuda()
        if pretrained:
            checkpoint = torch.load(sunet7128_path)
            sunet.load_state_dict(checkpoint['state_dict'])

        self.features = sunet.module.features

        for n, m in self.features.named_modules():
            if 'bn' in n:
                m.momentum = mom_bn

        # hack needed as batchnorm modules were not named as bn in residual block
        for n, m in self.features.residual1.conv.named_modules():
            if '2' in n:
                m.momentum = mom_bn

        self.final = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(2304, num_classes, kernel_size=1))
        ]))

        self.mceloss = cross_entropy2d(ignore=ignore_index, size_average=False, weight=weight)

    def forward(self, x, labels=None, th=1.0):
        x_size = x.size()
        x = self.features(x)
        x = F.relu(x, inplace=False)
        x = self.final(x)
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False)

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

# With degridding filters
class degrid_sunet7128(nn.Module):
    def __init__(self, num_classes, pretrained=True, ignore_index=-1, weight=None, output_stride='8'):
        super(degrid_sunet7128, self).__init__()
        self.num_classes = num_classes
        sunet = stackedunet7128(output_stride=output_stride)
        sunet = torch.nn.DataParallel(sunet, device_ids=range(torch.cuda.device_count())).cuda()
        if pretrained:
            checkpoint = torch.load(sunet7128_path)
            sunet.load_state_dict(checkpoint['state_dict'])

        self.features = sunet.module.features

        for n, m in self.features.named_modules():
            if 'bn' in n:
                m.momentum = mom_bn

        # hack needed as batchnorm modules were not named as bn in residual block
        for n, m in self.features.residual1.conv.named_modules():
            if '2' in n:
                m.momentum = mom_bn

        self.final = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(2304, 512, kernel_size=3, padding=2, dilation=2, bias=True)),
            ('bn1', nn.BatchNorm2d(512, momentum=mom_bn)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=True)),
            ('bn2', nn.BatchNorm2d(512, momentum=mom_bn)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(512, num_classes, kernel_size=1)),
        ]))

        self.mceloss = cross_entropy2d(ignore=ignore_index, size_average=False, weight=weight)

    def forward(self, x, labels=None, th=1.0):
        x_size = x.size()
        x = self.features(x)
        x = F.relu(x, inplace=False)
        x = self.final(x)
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False)

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