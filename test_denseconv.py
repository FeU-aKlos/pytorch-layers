import unittest
import torch 
from denseconv import _bn_function_factory,DenseLayer,DenseBlock,Transition
from math import ceil

class TestDenseLayer(unittest.TestCase):

    def setUp(self):
        self.in_channels = 3
        self.growth_rate = 12
        self.bn_size = 4
        self.efficient = True
        self.denseLayer = DenseLayer(
            in_channels=self.in_channels,
            growth_rate=self.growth_rate,
            bn_size=self.bn_size,
            efficient=self.efficient
        )
        self.batch_size = 16
        self.input_width = self.input_height = 32

    def test_forward(self):
        x = torch.rand([self.batch_size,self.in_channels,self.input_width,self.input_height])
        self.assertTrue(
            self.denseLayer(x).size()
                ==
            torch.Size(
                (
                    self.batch_size,self.growth_rate,self.input_width,self.input_height
                )
            )
        )


class TestTransition(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 16
        self.input_width=self.input_height = 32
        self.in_channels = 3
        self.growth_rate = 12
        self.bn_size = 4
        self.compression = 0.5
        self.out_channels = ceil((self.in_channels+self.bn_size*self.growth_rate)*self.compression)
        self.transition =Transition(self.in_channels, self.out_channels)

    def test_forward(self):
        x = torch.rand([self.batch_size,self.in_channels,self.input_width,self.input_height])
        self.assertTrue(
            self.transition(x).size()
                ==
            torch.Size(
                (
                    self.batch_size,self.out_channels,self.input_width//2,self.input_height//2
                )
            )
        )    

class TestDenseBlock(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.input_width=self.input_height = 32
        self.num_layers = 16
        self.in_channels = 3
        self.bn_size = 4
        self.growth_rate = 12
        self.efficient = False
        self.compression=0.5
        self.dense_block = DenseBlock(
            num_layers=self.num_layers,
            in_channels=self.in_channels,
            bn_size=self.bn_size,
            growth_rate=self.growth_rate,
            efficient=self.efficient,
            compression=self.compression
        )

    def test_forward(self):
        x = torch.rand([self.batch_size,self.in_channels,self.input_width,self.input_height])

        s = self.dense_block(x).size()
        out_size = int((self.in_channels+self.num_layers*self.growth_rate)*self.compression)

        self.assertTrue(
            s
                ==
            torch.Size(
                (
                    self.batch_size,out_size,self.input_width//2,self.input_height//2
                )
            )
        ) 