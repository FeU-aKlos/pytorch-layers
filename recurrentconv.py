import torch
from torch import nn

from baseconv import LayerBase, Conv2D_Layer
import config

class RecurrentOuput2HiddenWeightSharingConv(LayerBase):
    """Some Information about RecurrentConv
    https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liang_Recurrent_Convolutional_Neural_2015_CVPR_paper.pdf
    no localresponse normalization. After each layer bn
    no global max pooling
    """
    
    def __init__(self,in_channels, out_channels, kernel_size, stride=[1,1],padding=[0,0],steps=4):
        super(RecurrentOuput2HiddenWeightSharingConv, self).__init__()
        self.steps = steps
        self.conv_t0 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.conv_t = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=[kernel_size[0]-1,kernel_size[1]-1])
        #tied (shared) weights ;)
        self.conv_t.weight = self.conv_t0.weight
        self.initialize(self.conv_t)
        self.bn_conv_t =  nn.ModuleList([nn.BatchNorm2d(num_features=out_channels,momentum=config.batch_normalization_momentum) for i in range(steps)]) if config.employ_batch_normalization_conv else None
        self.dropout_conv_t = nn.ModuleList([nn.Dropout2d(p=1-config.dropout_rate) for i in range(steps)])if config.employ_dropout_conv else None
        self.scip_connection = Conv2D_Layer(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)

        self.is_bn = self.bn_conv_t!=None
        self.is_dropout = self.dropout_conv_t!=None

        self.act_function = self._activation_func()      

    def forward(self, x):
        x_s = self.scip_connection(x)
        x = self.conv_t0(x)
        if self.is_bn:
            x = self.bn_conv_t[0](x)
        x = self.act_function(x)
        if self.is_dropout:
            x = self.dropout_conv_t[0](x)
        for i in range(1,self.steps):
            x = self.conv_t(torch.add(x,x_s))
            if self.is_bn:
                x = self.bn_conv_t[i](x)
            x = self.act_function(x)
            if self.is_dropout:
                x = self.dropout_conv_t[i](x)
        return x
