# from https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import config

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, efficient=False):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = config.dropout_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if config.employ_dropout_conv:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class Transition(nn.Sequential):
    def __init__(self, in_channels, out_channles):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channles,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseBlock(nn.Module):
    r"""DenseBlock-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        num_layers (int) - determines, how many layers, the denseblock contains.
        in_channles (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        drop_rate (float) - dropout rate after each dense layer
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, num_layers, in_channels, bn_size=4, growth_rate=12, efficient=True, compression=0.5):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)
        out_channles = in_channels+num_layers * growth_rate
        trans = Transition(
            in_channels=out_channles, 
            out_channles=int(out_channles * compression)
        )
        self.add_module('transition%d' % (i + 1), trans)
        out_channles = int(out_channles * compression)


    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            if "transition" not in name:
                new_features = layer(*features)
                features.append(new_features)
            else:
                features = layer(torch.cat(features, 1))
        return features