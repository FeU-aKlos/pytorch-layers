import torch
from torch import is_grad_enabled, nn
from math import ceil
import torch.nn.functional as F
import config


class LayerBase(nn.Module):
    """
    @brief:
    This base class contains necessary parts for each feedforward layer type
    """
    def __init__(self, act_fn:str=config.act_function_name, is_last_layer:bool=False):
        super(LayerBase, self).__init__()
        self.is_last_layer = is_last_layer
        self.act_function_name = act_fn
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        @brief: should be newly implemented from each subclass
        """
        return x


    def _activation_func(self)->torch.Tensor:
        """
        @brief: returns the chosen torch activation function
        """
        activation_func_dict={
            'RELU': nn.ReLU,
            'SIGMOID': nn.Sigmoid,
            'TANH': nn.Tanh,
            'ELU': nn.ELU,
            'LEAKY_RELU':nn.LeakyReLU,
        }
        return activation_func_dict[self.act_function_name]()


    def _initialize(self,layer:torch.nn.Module):
        """
        @brief: initializes the weights depending on the activation function. ELU does not becomes not initialized by kaimung- or xavier uniform distribution, because its not supported by torch (current version 1.9)
        """
        if self.act_function_name in ["RELU",  "LEAKY_RELU"]:
            torch.nn.init.kaiming_uniform_(layer.weight,nonlinearity=self.act_function_name.lower())
        elif self.act_function_name in ["TANH",  "SIGMOID"]:
            torch.nn.init.xavier_uniform_(layer.weight,nn.init.calculate_gain(self.act_function_name.lower()))


class Conv2DBase(LayerBase):
    """
    @brief:
    This is the base class for conv2d. The activation function can be chosen. The Class inherits some basic methods from the parent class LayerBase. Furthermore apply normalization depending on the config. Besides, same padding can be calculated.
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size = [3,3],
            stride = [1,1],
            in_size=[32,32],
            employ_batch_normalization_conv = config.employ_batch_normalization_conv,
            employ_dropout_conv = config.employ_dropout_conv,
            act_function_name=config.act_function_name,
        ):
        super(Conv2DBase, self).__init__(act_function_name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.employ_batch_normalization_conv = employ_batch_normalization_conv
        self.employ_dropout_conv = employ_dropout_conv
        self.stride = stride
        self.in_size = in_size
        
        self.padding= self._calc_padding(in_size=in_size,kernel_size=kernel_size,stride=stride)
        self.add_module("same_padding",nn.ReplicationPad2d(self.padding) if self.padding!=0 or (len(self.padding)>1 and sum(self.padding)>=1) else None) 

    def forward(self, x:torch.Tensor)->torch.Tensor:

        return x

    def _apply_normalization(self,out_channels):
        """Provide batch normalization to layer"""
        return nn.BatchNorm2d(num_features=out_channels,momentum=config.batch_normalization_momentum)
    
    def _calc_padding(self,in_size,kernel_size,stride):
        if (in_size[0] % stride[0] == 0):
            pad_along_height = max(kernel_size[0] - stride[0], 0)
        else:
            pad_along_height = max(kernel_size[0] - (in_size[0] % stride[0]), 0)
        if (in_size[1] % stride[1] == 0):
            pad_along_width = max(kernel_size[1] - stride[1], 0)
        else:
            pad_along_width = max(kernel_size[1] - (in_size[1] % stride[1]), 0)
        
        #Finally, the padding on the top, bottom, left and right are:

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        return [pad_left,pad_right,pad_top,pad_bottom]

class Conv2D(Conv2DBase):
    """
    @brief:
    This is the base class for conv2d. Batchnormalization, dropout as well as the activation function can be chosen. The Class inherits some basic methods from the parent class LayerBase
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size = [3,3],
            stride = [1,1],
            in_size =  [32,32],
            employ_batch_normalization_conv = config.employ_batch_normalization_conv,
            employ_dropout_conv = config.employ_dropout_conv,
            act_function_name=config.act_function_name,
        ):
        super(Conv2D, self).__init__(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                employ_batch_normalization_conv=employ_batch_normalization_conv,
                employ_dropout_conv=employ_dropout_conv,
                in_size=in_size,
                stride=stride,
                act_function_name=act_function_name
            )
        
        
        self.add_module(
            "conv", nn.Conv2d(
                in_channels = in_channels, 
                out_channels = out_channels,
                kernel_size=kernel_size,
                bias = False if employ_batch_normalization_conv else True,
                stride = stride, 
                padding = [0,0]
            )
        )
        self._initialize(self.conv)
        
        self.add_module(
            "bn",
            self._apply_normalization(out_channels=out_channels) if employ_batch_normalization_conv else None
        )
        self.add_module(
            "act",
            self._activation_func()
        )
        self.add_module(
            "drop",
            nn.Dropout2d(p = 1 - config.dropout_rate) if employ_dropout_conv else None
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        if self.same_padding!=None:
            x = self.same_padding(x)
        x = self.conv(x)
        if self.bn!=None:
            x = self.bn(x)
        x = self.act(x)
        if self.drop!=None:
            x = self.drop(x)
        return x

class SampleConvNet(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels, 
            kernel_size, 
            stride, 
            in_size
        ):
        super(SampleConvNet, self).__init__()

        self.add_module(
            "conv",
            Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                in_size=in_size
            )
        )
        self.add_module("flatten",nn.Flatten())
        self.add_module("gap",nn.AdaptiveAvgPool2d((1,1)))
        self.add_module("fc",nn.Linear(out_channels, 10)) 
        

    def forward(self,x):
        x = self.conv(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output