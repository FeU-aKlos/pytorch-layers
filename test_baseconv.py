import unittest

from baseconv import Conv2D, LayerBase, Conv2DBase
import torch

from math import ceil

class TestLayerBase(unittest.TestCase):

    def setUp(self):
        self.layerbase_relu = LayerBase("RELU",False)
        self.module_list = torch.nn.ModuleList([self.layerbase_relu,LayerBase("SIGMOID",False),LayerBase("TANH",False),LayerBase("LEAKY_RELU",False),LayerBase("ELU",False)])        

    def test_activation_func(self):
        self.assertEqual(self.layerbase_relu.act_function_name,"RELU")
        self.assertEqual(self.layerbase_relu.is_last_layer,False)

        act_fn = self.layerbase_relu._activation_func()
        self.assertEqual(type(act_fn), torch.nn.ReLU)

    def test_initialize(self):
        for m in self.module_list:
            conv = torch.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=[2,2])
            w = conv.weight.clone()
            m._initialize(conv)
            is_equal = torch.equal(conv.weight,w)
            if m.act_function_name != "ELU":
                self.assertFalse(is_equal)
            else:
                self.assertTrue(is_equal)

class TestConv2DBase(unittest.TestCase):
    def setUp(self):
        
        self.conv2dbase2 = Conv2DBase(3,10)

    def test_apply_normalization(self):
        in_channels, out_channels = [1,20]
        self.conv2dbase = Conv2DBase(in_channels,out_channels)

        b, c, x, y = [2,20,9,9]
        rnd_input = torch.rand(size=[b,c,x,y])

        bn = self.conv2dbase._apply_normalization(out_channels)
        bn_out = bn(rnd_input)
        self.assertAlmostEqual(bn_out.mean().item(),0.0,places=1)
        self.assertAlmostEqual(bn_out.std().item(),1.0,places=1)

    def test_forward(self):
        in_channels, out_channels = [3,10]
        self.conv2dbase = Conv2DBase(in_channels,out_channels)
        b, c, x, y = [1,3,9,9]
        rnd_input = torch.rand(size=[b,c,x,y])
        output = self.conv2dbase(rnd_input)
        self.assertEqual(list(output.size()),[1,3,9,9])
        

class TestConv2D(unittest.TestCase):
    def setUp(self) -> None:
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
        self.conv2d = Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size, 
            stride=self.stride,
            in_size=[self.input_width,self.input_height]            
        )

    def test_forward(self):
        x = torch.rand([self.batch_size,self.in_channels,self.input_width,self.input_height])
        self.assertTrue(
            self.conv2d(x).size()
                ==
            torch.Size(
                (
                    self.batch_size,self.out_channels,self.input_width,self.input_height
                )
            )
        )
        
        self.stride=[3,3]
        self.conv2d = Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size, 
            stride=self.stride,
            in_size=[self.input_width,self.input_height]            
        )
        
        self.assertTrue(
            self.conv2d(x).size()
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
        self.conv2d = Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size, 
            stride=self.stride,
            in_size=[self.input_width,self.input_height]            
        )
        
        self.assertTrue(
            self.conv2d(x).size()
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



if __name__=="__main__":
    unittest.main()