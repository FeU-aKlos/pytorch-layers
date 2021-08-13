import unittest
import torch
from recurrentconv import RecurrentConv, SampleRecurrentConvNet
from math import ceil
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import config
from utils import train, test

class TestRecurrentConv(unittest.TestCase):

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
        self.recurrent_conv = RecurrentConv(
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
        self.recurrent_conv = RecurrentConv(
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
        self.recurrent_conv = RecurrentConv(
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

class TestSampleRecurrentConvNet(unittest.TestCase):
    def setUp(self):
        print()
        self.in_channels = 1
        self.out_channels = 32
        self.kernel_size = [3,3]
        self.stride = [2,2]
        self.in_size = [28,28]

        self.device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
        self.model = SampleRecurrentConvNet(
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



if __name__== "__main__":
    unittest.main()