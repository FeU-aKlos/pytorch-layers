import torch.nn as nn
import torch
import torch.nn.functional as F
from baseconv import Conv2DBase
import config

class ConvGRUFullyGated(Conv2DBase):
    """
    @brief: ConvGRUFullyGated inherrits from Conv2DBase class.
    more information can be obtain here: https://arxiv.org/pdf/1511.06432v4.pdf
    in_channels: int
        Number of channels of input tensor.
    hidden_channels: int
        Number of channels of hidden state.
    kernel_size: (int, int)
        Size of the convolutional kernel. Only uneven kernel sizes are currently supported! Usually, the kernel size is much smaller than the filter size (feature map width and height)
    stride: (int, int)
        stride of the convolutional kernel.
    in_size: (int, int)
        input width and height.
    employ_batch_normalization_conv: bool
        determines if batch normalization is employed for convolutional layers.
    time_steps: int
        How many timesteps should be applied.
    gru_type: str
        Valid values are FULLYGATED and TYPE2
    """
    def __init__(
            self, 
            in_channels, 
            hidden_channels, 
            kernel_size=[3,3], 
            stride=[1,1],
            in_size=[32,32],
            employ_batch_normalization_conv=config.employ_batch_normalization_conv,
            time_steps=4,
            gru_type="FULLYGATED"
        ):
        super(ConvGRUFullyGated, self).__init__(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            in_size=in_size,
            employ_batch_normalization_conv=employ_batch_normalization_conv
        )

        self.in_channels = in_channels  
        self.hidden_channels = hidden_channels

        self.kernel_size = kernel_size
        self.time_steps = time_steps
        self.gru_type = gru_type

        self.number_of_gates_and_cells = 3 if self.gru_type =="FULLYGATED" else 1


        self.padding_t = [kernel_size[0]//2,kernel_size[0]//2,kernel_size[1]//2,kernel_size[1]//2]
        self.add_module("same_padding_t",nn.ReplicationPad2d(self.padding_t)) 
        self.add_module(
            "conv_w",nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.number_of_gates_and_cells * self.hidden_channels,
                kernel_size=self.kernel_size,
                stride=[1,1],
                padding=[0,0],
                bias=False
            )
        )
        self._initialize(self.conv_w,"SIGMOID")
        self.add_module(
            "bn_w", 
            nn.ModuleList([
                nn.BatchNorm2d(
                    num_features=self.number_of_gates_and_cells*hidden_channels,
                    momentum=config.batch_normalization_momentum
                ) for i in range(self.time_steps)
            ])
        )
        self.add_module(
            "conv_uzr",nn.Conv2d(
                in_channels=self.hidden_channels,
                out_channels=2*self.hidden_channels,
                kernel_size=self.kernel_size,
                stride=[1,1],
                padding=[0,0],
                bias=False 
            )
        )
        self._initialize(self.conv_uzr,"SIGMOID")
        self.add_module(
            "conv_u",nn.Conv2d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                stride=[1,1],
                padding=[0,0],
                bias=False 
            )
        )
        self._initialize(self.conv_u,"SIGMOID")
        self.add_module(
            "bn_uzr",
            nn.ModuleList([
                nn.BatchNorm2d(
                    num_features=2*hidden_channels,
                    momentum=config.batch_normalization_momentum
                ) for i in range(self.time_steps-1)
            ])    
        )

        self.add_module(
            "bn_u",
            nn.ModuleList([
                nn.BatchNorm2d(
                    num_features=hidden_channels,
                    momentum=config.batch_normalization_momentum
                ) for i in range(self.time_steps-1)
            ]) 
        )


        self.add_module(
            "mp",nn.MaxPool2d(kernel_size=stride,stride=stride) if self.stride!=[1,1] else None
        )


        self.h_cur=None

        self.device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")


    def forward(self, input_tensor):
        #B,C,W,H
        if self.mp != None:
            input_tensor = self.mp(input_tensor)
        batch_size,_,width, height = input_tensor.size()
        
        self.h_cur = torch.zeros((batch_size,self.hidden_channels,width,height),dtype=torch.float).to(self.device)
                
        
        input_tensor=self.same_padding_t(input_tensor)
        input_tensor_conv = self.conv_w(input_tensor)
    
        f = self._forward_factory()
    
        f(input_tensor_conv)

        return self.h_cur

    def _forward_factory(self):
        if self.gru_type=="FULLYGATED":
            def fully_gated(x):
                for t in range(self.time_steps):
                    input_tensor_conv_bn=self.bn_w[t](x)
                    z, r, tmp_h = torch.chunk(input_tensor_conv_bn, self.number_of_gates_and_cells, dim=1)
                    if t>0:
                        h_cur_star = self.same_padding_t(self.h_cur)    
                        uz,ur = torch.chunk(self.bn_uzr[t-1](self.conv_uzr(h_cur_star)),2,dim=1)
                        z = torch.sigmoid(z+uz)
                        r = torch.sigmoid(r+ur)
                        tmp_h = torch.tanh(tmp_h+self.bn_u[t-1](self.conv_u(self.same_padding_t(r*self.h_cur))))
                        self.h_cur =(1-z)*self.h_cur+tmp_h*z
                    else:
                        z = torch.sigmoid(z)
                        r = torch.sigmoid(r)
                        self.h_cur = z*torch.tanh(tmp_h)
            return fully_gated
        elif self.gru_type=="TYPE2":
            def type2(x):
                for t in range(self.time_steps):
                    tmp_h=self.bn_w[t](x)
                    if t>0:
                        h_cur_star = self.same_padding_t(self.h_cur)    
                        uz,ur = torch.chunk(self.bn_uzr[t-1](self.conv_uzr(h_cur_star)),2,dim=1)
                        z = torch.sigmoid(uz)
                        r = torch.sigmoid(ur)
                        tmp_h = torch.tanh(tmp_h+self.bn_u[t-1](self.conv_u(self.same_padding_t(r*self.h_cur))))
                        self.h_cur =(1-z)*self.h_cur+tmp_h*z
                    else:
                        z = torch.sigmoid(self.h_cur)
                        r = torch.sigmoid(self.h_cur)
                        self.h_cur = z*torch.tanh(tmp_h)
            return type2




class SampleConvGRUFullyGatedNet(nn.Module):
    """
    @brief: Sample Network demonstrating the utilization of a ConvLSTM layer.
    """
    def __init__(self,in_channels,hidden_channels, kernel_size, stride, in_size):
        super(SampleConvGRUFullyGatedNet, self).__init__()

        self.add_module(
            "convgru_fully_gated",
            ConvGRUFullyGated(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                in_size=in_size
            )
        )
        self.add_module("gap",nn.AdaptiveAvgPool2d((1,1)))
        self.add_module("flatten",nn.Flatten())
        self.add_module("fc",nn.Linear(hidden_channels, 10)) 
        

    def forward(self,x):
        x = self.convgru_fully_gated(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class SampleConvGRUTye2Net(nn.Module):
    """
    @brief: Sample Network demonstrating the utilization of a ConvLSTM layer.
    """
    def __init__(self,in_channels,hidden_channels, kernel_size, stride, in_size):
        super(SampleConvGRUTye2Net, self).__init__()

        self.add_module(
            "convgru_fully_gated",
            ConvGRUFullyGated(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                in_size=in_size,
                gru_type="TYPE2"
            )
        )
        self.add_module("gap",nn.AdaptiveAvgPool2d((1,1)))
        self.add_module("flatten",nn.Flatten())
        self.add_module("fc",nn.Linear(hidden_channels, 10)) 
        

    def forward(self,x):
        x = self.convgru_fully_gated(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

