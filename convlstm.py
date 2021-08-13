import torch.nn as nn
import torch
import torch.nn.functional as F
from baseconv import Conv2DBase
import config

class ConvLSTM(Conv2DBase):
    """
    @brief: ConvLSTM inherrits from Conv2DBase class.
    in_channels: int
        Number of channels of input tensor.
    hidden_channels: int
        Number of channels of hidden state.
    kernel_size: (int, int)
        Size of the convolutional kernel. Only uneven kernel sizes are currently supported!
    stride: (int, int)
        stride of the convolutional kernel.
    in_size: (int, int)
        input width and height.
    employ_batch_normalization_conv: bool
        determines if batch normalization is employed for convolutional layers.
    time_steps: int
        How many timesteps should be applied.
    """
    def __init__(
            self, 
            in_channels, 
            hidden_channels, 
            kernel_size=[3,3], 
            stride=[1,1],
            in_size=[32,32],
            employ_batch_normalization_conv=config.employ_batch_normalization_conv,
            time_steps=4
        ):
        super(ConvLSTM, self).__init__(
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

        self.padding_t = [kernel_size[0]//2,kernel_size[0]//2,kernel_size[1]//2,kernel_size[1]//2]
        self.add_module("same_padding_t",nn.ReplicationPad2d(self.padding_t) if self.padding_t!=0 or (len(self.padding_t)>1 and sum(self.padding_t)>=1) else None) 
        self.add_module(
            "conv_t",nn.Conv2d(
                in_channels=self.in_channels + self.hidden_channels,
                out_channels=4 * self.hidden_channels,
                kernel_size=self.kernel_size,
                stride=[1,1],
                padding=[0,0],
                bias=False if config.employ_batch_normalization_conv else True
            )
        )
        
        self.add_module(
            "bn", nn.ModuleList(
                [
                    nn.BatchNorm2d(
                        num_features=4* hidden_channels,
                        momentum=config.batch_normalization_momentum
                    ) 
                    for i in range(time_steps)
                ]) if config.employ_batch_normalization_conv else None
        )

        self.add_module(
            "mp",nn.MaxPool2d(kernel_size=stride,stride=stride) if self.stride!=[1,1] else None
        )

        self.Wci = None
        self.Wcf = None
        self.Wco = None

        self.number_of_gates_and_cells = 4

        self.device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")


    def forward(self, input_tensor):
        #B,C,W,H
        if self.mp != None:
            input_tensor = self.mp(input_tensor)
        batch_size,_,width, height = input_tensor.size()
        
        #maybe to cuda o.O
        c_cur = torch.zeros((batch_size,self.hidden_channels,width,height),dtype=torch.float).to(self.device)
        h_cur = torch.zeros((batch_size,self.hidden_channels,width,height),dtype=torch.float).to(self.device)
            
        for t in range(self.time_steps):            
            combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
            if self.same_padding_t!=None:
                combined=self.same_padding_t(combined)
            combined_conv = self.conv_t(combined)
            if self.bn!=None:
                combined_conv=self.bn[t](combined_conv)
            i, f, tmp_c, o = torch.chunk(combined_conv, self.number_of_gates_and_cells, dim=1)
            if t>0:
                i = torch.sigmoid(i+self.Wci*c_cur)
                f = torch.sigmoid(f+self.Wcf*c_cur)
                c_cur = f*c_cur+i* torch.tanh(tmp_c)
                o = torch.sigmoid(o+self.Wco*c_cur)
            else:
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                c_cur = f*c_cur+i* torch.tanh(tmp_c)
                o = torch.sigmoid(o)
                
                self.initialize_gates(width=width,height=height)
            
            h_cur = o * torch.tanh(c_cur)

        return o

    def initialize_gates(self, width,height):
        self.Wci = nn.Parameter(
            torch.zeros(
                1, 
                self.hidden_channels, 
                width, 
                height
            )
        ).to(self.device)
        self.Wcf = nn.Parameter(
            torch.zeros(
                1, 
                self.hidden_channels, 
                width, 
                height
            )
        ).to(self.device)
        self.Wco = nn.Parameter(
            torch.zeros(
                1, 
                self.hidden_channels, 
                width, 
                height
            )
        ).to(self.device)

class SampleConvLSTMNet(nn.Module):
    """
    @brief: Sample Network demonstrating the utilization of a ConvLSTM layer.
    """
    def __init__(self,in_channels,hidden_channels, kernel_size, stride, in_size):
        super(SampleConvLSTMNet, self).__init__()

        self.add_module(
            "convlstm",
            ConvLSTM(
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
        x = self.convlstm(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output