import unittest

from baseconv import Conv2D, LayerBase, Conv2DBase, SampleConvNet
import config
from utils import test,train
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch

from math import ceil

class TestLayerBase(unittest.TestCase):

    def setUp(self):
        self.layerbase_relu = LayerBase("RELU",False)
        self.module_list = torch.nn.ModuleList([self.layerbase_relu,LayerBase("SIGMOID",False),LayerBase("TANH",False),LayerBase("LEAKY_RELU",False)])        

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
            self.assertFalse(is_equal)


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

class TestSampleConvNet(unittest.TestCase):
    def setUp(self):
        print()
        self.in_channels = 1
        self.out_channels = 32
        self.kernel_size = [3,3]
        self.stride = [2,2]
        self.in_size = [28,28]

        self.device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
        self.model = SampleConvNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            in_size=self.in_size
        ).to(self.device)
        torch.manual_seed(config.seed)


        train_kwargs = {'batch_size': config.batch_size}
        test_kwargs = {'batch_size': config.test_batch_size}
        if self.device.type=="cuda":
            cuda_kwargs = {'num_workers': 0,
                        'pin_memory': True,
                        'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            cuda_kwargs = {'num_workers': 0,
                        'pin_memory': True,
                        'shuffle': False}
            test_kwargs.update(cuda_kwargs)

        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
        dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


    def test_train_and_test(self):

        optimizer = optim.Adadelta(self.model.parameters(), lr=config.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
        for epoch in range(1, config.epochs + 1):
            train(self.model, self.device, self.train_loader, optimizer, epoch)
            scheduler.step()
            print()
        test(self.model, self.device, self.test_loader)


if __name__=="__main__":
    unittest.main()