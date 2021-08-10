from datetime import time
import unittest
from lstmconv import ConvLSTM
import torch
class TestConvLSTM(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 16
        self.input_width = self.input_height=32
        self.in_channels = 3
        self.out_channels = self.in_channels
        self.hidden_channels = 10
        self.kernel_size = [3,3]
        self.bias = False
        self.time_steps = 5
        self.conv_lstm = ConvLSTM(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            bias=self.bias,
            time_steps=self.time_steps
        )
    
    def test_forward(self):
        x = torch.rand([self.batch_size,self.in_channels,self.input_width,self.input_height])
        self.assertTrue(
            self.conv_lstm(x).size()
                ==
            torch.Size(
                (
                    self.batch_size,self.hidden_channels,self.input_width,self.input_height
                )
            )
        )    

    def test_initialize(self):
        self.conv_lstm = ConvLSTM(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            bias=self.bias,
            time_steps=self.time_steps
        )

        self.assertEqual(self.conv_lstm.Wci,None)
        self.assertEqual(self.conv_lstm.Wcf,None)
        self.assertEqual(self.conv_lstm.Wco,None)
        
        self.conv_lstm.initialize(self.input_width,self.input_height)
        
        self.assertNotEqual(self.conv_lstm.Wci,None)
        self.assertNotEqual(self.conv_lstm.Wcf,None)
        self.assertNotEqual(self.conv_lstm.Wco,None)

        self.assertTrue(self.conv_lstm.Wci.requires_grad)
        self.assertTrue(self.conv_lstm.Wcf.requires_grad)
        self.assertTrue(self.conv_lstm.Wco.requires_grad)

        self.assertTrue(
            self.conv_lstm.Wci.size()
                ==
            torch.Size(
                (
                    1,self.hidden_channels,self.input_width,self.input_height
                )
            )
        ) 
        self.assertTrue(
            self.conv_lstm.Wcf.size()
                ==
            torch.Size(
                (
                    1,self.hidden_channels,self.input_width,self.input_height
                )
            )
        ) 
        self.assertTrue(
            self.conv_lstm.Wco.size()
                ==
            torch.Size(
                (
                    1,self.hidden_channels,self.input_width,self.input_height
                )
            )
        ) 
    
