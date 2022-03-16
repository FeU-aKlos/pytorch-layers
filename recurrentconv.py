import torch
from torch import nn

from baseconv import Conv2DBase,Conv2D
import torch.nn.functional as F
import config

class RecurrentConv(Conv2DBase):
    """
    @bief: RecurrentConv inherrits from Conv2DBase
    Imformation about the network can be found in
    https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liang_Recurrent_Convolutional_Neural_2015_CVPR_paper.pdf
    Differing from the paper instead of localresponse normalization batch normalization is applied. No global max pooling
    in_channels (int): Determines the number of input channels
    out_channels (int): Depicts the number of output channels
    kernel size [int,int]: Is the kernel size
    stride [int,int]: the stride
    in_size [int,int]: the input width and height
    employ_batch_normalization_conv (bool): if batch normalization should be applied
    employ_dropout_conv (bool): if dropout should be applied
    steps (int): how many time steps
    """
    def __init__(
            self,
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=[1,1],
            in_size=[32,32],
            employ_batch_normalization_conv=config.employ_batch_normalization_conv,
            employ_dropout_conv=config.employ_dropout_conv,
            steps=4
        ):
        super(RecurrentConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            in_size=in_size,
            employ_batch_normalization_conv=employ_batch_normalization_conv,
            employ_dropout_conv=employ_dropout_conv
        )
        self.steps = steps
                
        self.add_module(
            "same_padding_hidden",
            nn.ReplicationPad2d(
                [
                    self.kernel_size[0]//2,
                    self.kernel_size[0]//2-(0 if self.kernel_size[0]%2!=0 else 1),
                    self.kernel_size[1]//2,
                    self.kernel_size[1]//2-(0 if self.kernel_size[1]%2!=0 else 1)
                ]
            ) 
        )
        self.add_module(
            "conv_t",
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=[0,0]
            )
        ) 

        self._initialize(self.conv_t)
        
        self.add_module(
            "bn_conv_t", 
            nn.ModuleList(
                [
                    nn.BatchNorm2d(
                        num_features=out_channels,
                        momentum=config.batch_normalization_momentum
                    ) for i in range(steps)
                ]
            ) if self.employ_batch_normalization_conv else None
        )
        
        self.add_module(
            "dropout_conv_t",
            nn.ModuleList(
                [
                    nn.Dropout2d(
                        p=1-config.dropout_rate
                    ) for i in range(steps)
                ]
            )if config.employ_dropout_conv else None
        )
        
        self.add_module(
            "scip_connection",
            Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                in_size=in_size
            )
        )
        self._initialize(self.scip_connection.conv)

        self.is_bn = self.bn_conv_t!=None
        self.is_dropout = self.dropout_conv_t!=None

        self.add_module(
            "act_function",
            self._activation_func()
        )

    def forward(self, x):
        x_s = self.scip_connection(x)
 
        x = self.same_padding_hidden(x_s)
        x = self.conv_t(x)
        if self.is_bn:
            x = self.bn_conv_t[0](x)
        x = self.act_function(x)
        if self.is_dropout:
            x = self.dropout_conv_t[0](x)
        for i in range(1,self.steps):
            x = torch.add(x,x_s)
            x = self.same_padding_hidden(x)
            x = self.conv_t(x)
            if self.is_bn:
                x = self.bn_conv_t[i](x)
            x = self.act_function(x)
            if self.is_dropout:
                x = self.dropout_conv_t[i](x)
        return x

class SampleRecurrentConvNet(nn.Module):
    """
    @brief: Sample Network demonstrating the utilization of a RecurrentConv layer.
    """
    def __init__(
            self,
            in_channels,
            out_channels, 
            kernel_size, 
            stride, 
            in_size
        ):
        super(SampleRecurrentConvNet, self).__init__()

        self.add_module(
            "rc",
            RecurrentConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                in_size=in_size
            )
        )
        self.add_module("gap",nn.AdaptiveAvgPool2d((1,1)))
        self.add_module("flatten",nn.Flatten())
        self.add_module("fc",nn.Linear(out_channels, 10)) 
        

    def forward(self,x):
        x = self.rc(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output