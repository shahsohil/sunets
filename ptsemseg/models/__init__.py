import torch.nn as nn
from torch.nn import init

from resnet import *
from sunet import *

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias.data is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            if m.bias.data is not None:
                init.constant_(m.bias, 0)

def get_model(name, n_classes, ignore_index=-1, weight=None, output_stride='16'):
    model = _get_model_instance(name)

    if name == 'resnet18' or name == 'resnet101' or name == 'sunet64' or name == 'sunet128' or name == 'sunet7128' or name == 'degridsunet7128':
        model = model(num_classes=n_classes, ignore_index=ignore_index, weight=weight, output_stride=output_stride)
        init_params(model.final)
    else:
        raise 'Model {} not available'.format(name)

    return model

def _get_model_instance(name):
    return {
        'resnet18': d_resnet18,
        'resnet101': d_resnet101,
        'sunet64': d_sunet64,
        'sunet128': d_sunet128,
        'sunet7128': d_sunet7128,
        'degridsunet7128': degrid_sunet7128,
    }[name]
