import torch.nn as nn
import torch
import math
import collections
import numpy as np

import torch.utils.model_zoo as model_zoo

__all__ = [
    'ResNet', 
    'flow_resnet18', 
    'flow_resnet34', 
    'flow_resnet50', 
    'flow_resnet50_aux', 
    'flow_resnet101',
    'flow_resnet152'
]


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, num_segments):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(20, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        # self.fc_aux = nn.Linear(512 * block.expansion, 101)
        self.dp = nn.Dropout(p=0.5)
        self.fc_action = nn.Linear(512 * block.expansion, num_classes)

        self.num_segments = num_segments
        # self.bn_final = nn.BatchNorm1d(num_classes)
        # self.fc2 = nn.Linear(num_classes, num_classes)
        # self.fc_final = nn.Linear(num_classes, 101)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        # segmental consensus
        x = torch.reshape(x, (-1, self.num_segments, 512))
        x = torch.mean(x,dim = 1)

        x = self.fc_action(x)
        # x = self.bn_final(x)
        # x = self.fc2(x)
        # x = self.fc_final(x)

        return x

def change_key_names(old_params, in_channels):
    new_params = collections.OrderedDict()
    layer_count = 0
    allKeyList = old_params.keys()
    for layer_key in allKeyList:
        if layer_count >= len(allKeyList)-2:
            # exclude fc layers
            continue
        else:
            if layer_count == 0:
                rgb_weight = old_params[layer_key]
                # print(type(rgb_weight))
                rgb_weight_mean = torch.mean(rgb_weight, dim=1)
                # TODO: ugly fix here, why torch.mean() turn tensor to Variable
                # print(type(rgb_weight_mean))
                flow_weight = rgb_weight_mean.unsqueeze(1).repeat(1,in_channels,1,1)
                new_params[layer_key] = flow_weight
                layer_count += 1
                # print(layer_key, new_params[layer_key].size(), type(new_params[layer_key]))
            else:
                new_params[layer_key] = old_params[layer_key]
                layer_count += 1
                # print(layer_key, new_params[layer_key].size(), type(new_params[layer_key]))
    
    return new_params

def flow_resnet18(num_classes, num_segments, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, num_segments, **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        in_channels = 20
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()

        new_pretrained_dict = change_key_names(pretrained_dict, in_channels)
        # 1. filter out unnecessary keys
        new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model

def flow_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        model_dict = model.state_dict()

        new_pretrained_dict = change_key_names(pretrained_dict, in_channels)

        # 1. filter out unnecessary keys
        new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model

def flow_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    in_channels = 20
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])

        model_dict = model.state_dict()

        new_pretrained_dict = change_key_names(pretrained_dict, in_channels)

        # 1. filter out unnecessary keys
        new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model

def flow_resnet50_aux(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])

        model_dict = model.state_dict()
        fc_origin_weight = pretrained_dict["fc.weight"].data.numpy()
        fc_origin_bias = pretrained_dict["fc.bias"].data.numpy()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # print(model_dict)
        fc_new_weight = model_dict["fc_aux.weight"].numpy() 
        fc_new_bias = model_dict["fc_aux.bias"].numpy() 

        fc_new_weight[:1000, :] = fc_origin_weight
        fc_new_bias[:1000] = fc_origin_bias

        model_dict["fc_aux.weight"] = torch.from_numpy(fc_new_weight)
        model_dict["fc_aux.bias"] = torch.from_numpy(fc_new_bias)

        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model

def flow_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

def flow_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        in_channels = 20
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
        pretrained_dict = model_zoo.load_url(model_urls['resnet152'])
        model_dict = model.state_dict()

        new_pretrained_dict = change_key_names(pretrained_dict, in_channels)
        # 1. filter out unnecessary keys
        new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model