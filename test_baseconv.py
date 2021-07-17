import unittest

from baseconv import LayerBase, Conv2DBase
import torch

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

        bn_out = self.conv2dbase.bn(rnd_input)
        self.assertAlmostEqual(bn_out.mean().item(),0.0,places=1)
        self.assertAlmostEqual(bn_out.std().item(),1.0,places=1)

    def test_forward(self):
        in_channels, out_channels = [3,10]
        self.conv2dbase = Conv2DBase(in_channels,out_channels)
        b, c, x, y = [1,3,9,9]
        rnd_input = torch.rand(size=[b,c,x,y])
        output = self.conv2dbase(rnd_input)
        self.assertEqual(list(output.size()),[1,10,7,7])
        
        kernel_size = [3,3]
        padding = [(kernel_size[0]-1)//2,(kernel_size[0]-1)//2,(kernel_size[1]-1)//2,(kernel_size[1]-1)//2]
        self.conv2dbase = Conv2DBase(in_channels,out_channels,kernel_size=kernel_size,padding=padding)

        output = self.conv2dbase(rnd_input)
        self.assertEqual(list(output.size()),[b,out_channels,x,y])

        





if __name__=="__main__":
    unittest.main()