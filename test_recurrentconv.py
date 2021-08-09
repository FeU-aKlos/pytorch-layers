import unittest
import torch
from recurrentconv import RecurrentOuput2HiddenWeightSharingConv
from math import ceil

class TestRecurrentOuput2HiddenWeightSharingConv(unittest.TestCase):

    def setUp(self):
        self.in_channels = 3
        self.out_channels=10
        self.kernel_size = [3,3]
        self.padding = [
            self.kernel_size[0]//2,
            self.kernel_size[0]//2,
            self.kernel_size[1]//2,
            self.kernel_size[1]//2
        ]
        self.stride = [1,1]
        self.batch_size = 16
        self.input_width = self.input_height=32
        self.recurrent_conv = RecurrentOuput2HiddenWeightSharingConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size, 
            stride=self.stride,
            in_size=[self.input_width,self.input_height]            
        )


    def test_forward(self):
        
        x = torch.rand([self.batch_size,self.in_channels,self.input_width,self.input_height])
        self.assertTrue(
            self.recurrent_conv(x).size()
                ==
            torch.Size(
                (
                    self.batch_size,self.out_channels,self.input_width,self.input_height
                )
            )
        )
        
        self.stride=[3,3]
        self.recurrent_conv = RecurrentOuput2HiddenWeightSharingConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size, 
            stride=self.stride,
            in_size=[self.input_width,self.input_height]            
        )
        
        self.assertTrue(
            self.recurrent_conv(x).size()
            ==
            torch.Size(
                (
                    self.batch_size,
                    self.out_channels,
                    ceil(self.input_width/self.stride[0]),
                    ceil(self.input_height/self.stride[1])
                )
            )
        )

        self.kernel_size=[7,7]
        self.recurrent_conv = RecurrentOuput2HiddenWeightSharingConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size, 
            stride=self.stride,
            in_size=[self.input_width,self.input_height]            
        )
        
        self.assertTrue(
            self.recurrent_conv(x).size()
            ==
            torch.Size(
                (
                    self.batch_size,
                    self.out_channels,
                    ceil(self.input_width/self.stride[0]),
                    ceil(self.input_height/self.stride[1])
                )
            )
        )


if __name__== "__main__":
    unittest.main()