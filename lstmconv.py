import torch.nn as nn
import torch

import config

class ConvLSTM(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, bias,time_steps):
        """
        Initialize ConvLSTM.

        Parameters
        ----------
        in_channels: int
            Number of channels of input tensor.
        hidden_channels: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2#adjust padding ;) and kernel size...
        self.bias = bias
        self.time_steps = time_steps

        self.add_module(
            "conv",nn.Conv2d(
                in_channels=self.in_channels + self.hidden_channels,
                out_channels=4 * self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=self.bias
            )
        )
        self.add_module(
            "bn", nn.ModuleList(
                [
                    nn.BatchNorm2d(
                        num_features=hidden_channels,
                        momentum=config.batch_normalization_momentum
                    ) 
                    for i in range(time_steps)
                ]) if config.employ_batch_normalization_conv else None
        )

        self.Wci = None
        self.Wcf = None
        self.Wco = None

        self.number_of_gates_and_cells = 4


    def forward(self, input_tensor):
        #B,C,W,H
        batch_size,_,width, height = input_tensor.size()
        
        #maybe to cuda o.O
        c_cur = torch.zeros((batch_size,self.hidden_channels,width,height),dtype=torch.float)
        h_cur = torch.zeros((batch_size,self.hidden_channels,width,height),dtype=torch.float)
                
        for t in range(self.time_steps):
                
            combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

            combined_conv = self.conv(combined)
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
                
                self.initialize(batch_size=batch_size,width=width,height=height)
            
            h_cur = o * torch.tanh(c_cur)

        return o

    def initialize(self, width,height):
        self.Wci = nn.Parameter(
            torch.zeros(
                1, 
                self.hidden_channels, 
                width, 
                height
            )
        )
        self.Wcf = nn.Parameter(
            torch.zeros(
                1, 
                self.hidden_channels, 
                width, 
                height
            )
        )
        self.Wco = nn.Parameter(
            torch.zeros(
                1, 
                self.hidden_channels, 
                width, 
                height
            )
        )