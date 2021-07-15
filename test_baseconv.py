import unittest
from baseconv import LayerBase
import torch

class TestBaseConv(unittest.TestCase):

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





if __name__=="__main__":
    unittest.main()