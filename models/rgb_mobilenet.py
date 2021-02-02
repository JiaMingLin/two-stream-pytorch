from torch import nn
import torch.utils.model_zoo as model_zoo
import os
import math
import collections
import numpy as np
import torch
#from .utils import load_state_dict_from_url
#from torch.hub import load_state_dict_from_url
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['MobileNetV2', 'mobilenet_v2', 'rgb_mobilenet']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 num_segments = 3):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2, self).__init__()

        self.num_segments = num_segments

        if block is None:
            block = InvertedResidual
        input_channel = 32 #32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.last_channel, num_classes),

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)

        x = self.dropout(x)

        # segmental consensus
        x = torch.reshape(x, (-1, self.num_segments, self.last_channel))
        x = torch.mean(x,dim = 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def mobilenet_v2(pretrained=False, progress=True, num_segments=3, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def modified_model_dict_key(old_params):
    '''
    Parameters
    ----------
    old_params : old key name of state_dict from now mobilenet format
    
    Returns
    -------
    new_params : new modified key name which is same with now mobilenet format

    '''
    new_params = collections.OrderedDict()
    with open (os.path.abspath(__file__+'/../../')+'/models/modified_state_key_MobileNetv2.txt', 'r') as file:
        for line in file:
            oldkey, newkey = line.strip('\n').split(' ')
            new_params[newkey] = old_params[oldkey]
    return new_params

# f = open('modified_state_key_MobileNetv2.txt', 'w')
# for (k1,v1), (k2,v2) in zip(pretrained_dict.items(), model_dict.items()):
   
#     if (v1.shape == v2.shape):
#         print(k1,k2)
#         print(k1,k2,file=f)
# f.close()

def rgb_mobilenet(num_classes, pretrained=False, progress=True, num_segments=3, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(num_classes = num_classes, num_segments=num_segments, **kwargs)
    
    if pretrained:
        if 'width_mult' not in kwargs.keys():
            pretrained_dict = model_zoo.load_url(model_urls['mobilenet_v2'])
            if 'classifier.1.weight' in pretrained_dict: del pretrained_dict['classifier.1.weight']
            if 'classifier.1.bias' in pretrained_dict: del pretrained_dict['classifier.1.bias']
            print('Load mobilenet_v2 pretrained model, remove the last classifier layer for new num_classes')
        else:
            model_path = os.path.abspath(__file__+'/../../')+'/checkpoints/mobilenetv2_pretrained/mobilenetv2_{}.pth'.format(kwargs['width_mult'])
            pretrained_dict = torch.load(model_path)
            pretrained_dict = modified_model_dict_key(pretrained_dict)
            print('Load model from {}'.format(model_path))
            # special case for width_mult 0.1 pretrained model
            if (kwargs['width_mult'] == 0.1):
                model = MobileNetV2(num_classes=101, width_mult=0.1, round_nearest=4)              
        
        model_dict = model.state_dict()
       
        # 1. filter out unnecessary keys
        new_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model

# if __name__ == '__main__':
     #width_mult = 0.1
     #model = rgb_mobilenet(pretrained=True, num_classes=101, width_mult = width_mult)
