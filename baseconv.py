import torch
from torch import is_grad_enabled, nn
from math import ceil

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
    This is the base class for conv2d. Batchnormalization, dropout as well as the activation function can be chosen. The Class inherits some basic methods from the parent class LayerBase
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size = [3,3],
            employ_batch_normalization_conv = config.employ_batch_normalization_conv,
            employ_dropout_conv = config.employ_dropout_conv,
            padding = [0,0],
            stride = [1,1],
            act_function_name=config.act_function_name,
        ):
        super(Conv2DBase, self).__init__(act_function_name)
        
        self.same_padding = nn.ReplicationPad2d(padding) if padding!=[0,0] else None
        self.conv = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = out_channels,
            kernel_size=kernel_size,
            bias = False if employ_batch_normalization_conv else True,
            stride = stride, 
            padding = [0,0]
        )
        self._initialize(self.conv)
        
        self.bn = self._apply_normalization(out_channels=out_channels) if employ_batch_normalization_conv else None
        self.act = self._activation_func()
        self.drop = nn.Dropout2d(p = 1 - config.dropout_rate) if employ_dropout_conv else None

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

    def _apply_normalization(self,out_channels):
        """Provide batch normalization to layer"""
        return nn.BatchNorm2d(num_features=out_channels,momentum=config.batch_normalization_momentum)

# class InceptionBranchFactorization1(nn.Module):
#     """Some Information about InceptionBranchFactorization1"""
#     def __init__(self,in_channels, out_channels, kernel_size, stride, in_size):
#         super(InceptionBranchFactorization1, self).__init__()

#         self.branch_1 = ConvLayerBase(in_channels, out_channels, kernel_size=1)
#         ks=(ceil(kernel_size[0]/2), ceil(kernel_size[1]/2))
#         padding = [
#             ks[1]//2-1+ks[1]%2,
#             ks[1]//2,
#             ks[0]//2-1+ks[0]%2,
#             ks[0]//2
#         ]
#         self.same_padding1 = nn.ReplicationPad2d(padding)

#         self.branch_2 = ConvLayerBase(out_channels, out_channels, kernel_size=ks)
#         padding = calc_padding([in_size[1],in_size[2]],ks,stride)
#         self.same_padding2 = nn.ReplicationPad2d(padding)
#         self.branch_3 = ConvLayerBase(out_channels, out_channels, kernel_size=ks, stride=stride)

#     def forward(self, x):
#         x = self.branch_1(x)
#         x = self.same_padding1(x)
#         x = self.branch_2(x)
#         x = self.same_padding2(x)
#         x = self.branch_3(x)

#         return x

# class InceptionBranchFactorization1Split(nn.Module):
#     """Some Information about InceptionBranchFactorization1Split"""
#     def __init__(self,in_channels, out_channels, kernel_size, stride,in_size):
#         super(InceptionBranchFactorization1Split, self).__init__()

#         self.branch_1 = ConvLayerBase(in_channels, out_channels, kernel_size=1)
#         ks=(ceil(kernel_size[0]/2), ceil(kernel_size[1]/2))
#         padding = [
#             ks[1]//2-1+ks[1]%2,
#             ks[1]//2,
#             ks[0]//2-1+ks[0]%2,
#             ks[0]//2
#         ]
#         self.same_padding_1 = nn.ReplicationPad2d(padding)
#         self.branch_2 = ConvLayerBase(out_channels, out_channels, kernel_size=ks)
#         ks=(1, ceil(kernel_size[1]/2))
#         padding = calc_padding([in_size[1],in_size[2]],ks,stride)
#         self.same_padding_2 = nn.ReplicationPad2d(padding)
#         self.branch_3a = ConvLayerBase(out_channels, out_channels, kernel_size=ks, stride=stride)
#         ks=(ceil(kernel_size[0]/2),1)
#         padding = calc_padding([in_size[1],in_size[2]],ks,stride)
#         self.same_padding_3 = nn.ReplicationPad2d(padding)
#         self.branch_3b = ConvLayerBase(out_channels, out_channels, kernel_size=ks, stride=stride)

#     def forward(self, x):
#         x = self.branch_1(x)
#         x = self.same_padding_1(x)
#         x = self.branch_2(x)
#         xa = self.same_padding_2(x)
#         xa = self.branch_3a(xa)
#         xb = self.same_padding_3(x)
#         xb = self.branch_3b(xb)


#         return torch.cat([xa,xb],1)

# class InceptionBranchAsynchFactorization(nn.Module):
#     """Some Information about InceptionBranchAsynchFactorization"""
#     def __init__(self,in_channels, out_channels, kernel_size, stride,in_size):
#         super(InceptionBranchAsynchFactorization, self).__init__()
        
#         self.branch_1 = ConvLayerBase(in_channels, out_channels, kernel_size=1)
#         ks = (1, kernel_size[1])
#         s = (1,stride[1])
#         padding = calc_padding([in_size[1],in_size[2]],ks,s)
        
#         self.same_padding_1 = nn.ReplicationPad2d(padding)
#         self.branch_2 = ConvLayerBase(out_channels, out_channels, kernel_size=(1, kernel_size[1]), stride=(1,stride[1]))
#         ks = (kernel_size[0],1)
#         s = (stride[0],1)
#         padding = calc_padding([in_size[1],in_size[2]],ks,s)
#         self.same_padding_2 = nn.ReplicationPad2d(padding)
#         self.branch_3 = ConvLayerBase(out_channels, out_channels, kernel_size=(kernel_size[0], 1), stride=(stride[0],1))

#     def forward(self, x):
#         x = self.branch_1(x)
#         x = self.same_padding_1(x)
#         x = self.branch_2(x)
#         x = self.same_padding_2(x)
#         x = self.branch_3(x)

#         return x

def calc_padding(in_size,kernel_size,stride):
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