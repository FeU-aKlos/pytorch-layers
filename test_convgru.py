import torch
from datetime import time
import unittest
from convgru import ConvGRUFullyGated,SampleConvGRUFullyGatedNet,SampleConvGRUTye2Net
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import config
from utils import train, test

class TestConvGRUFullyGated(unittest.TestCase):
    
    def setUp(self):
        self.device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
        self.batch_size = 16
        self.input_width = self.input_height=32
        self.in_channels = 3
        self.out_channels = self.in_channels
        self.hidden_channels = 10
        self.kernel_size = [3,3]
        self.stride = [1,1]
        self.bias = False
        self.time_steps = 5
        self.conv_gru = ConvGRUFullyGated(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            in_size=[self.input_width,self.input_height],
            time_steps=self.time_steps
        ).to(self.device)
    
    def test_forward(self):
        x = torch.rand([self.batch_size,self.in_channels,self.input_width,self.input_height]).to(self.device)
        self.assertTrue(
            self.conv_gru(x).size()
                ==
            torch.Size(
                (
                    self.batch_size,self.hidden_channels,self.input_width,self.input_height
                )
            )
        )    
        
        stride = [2,3]
        self.conv_gru = ConvGRUFullyGated(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            in_size=[self.input_width,self.input_height],
            time_steps=self.time_steps
        ).to(self.device)

        self.assertTrue(
            self.conv_gru(x).size()
                ==
            torch.Size(
                (
                    self.batch_size,self.hidden_channels,self.input_width//stride[0],self.input_height//stride[1]
                )
            )
        )  

        kernel_size=[7,7]
        self.conv_gru = ConvGRUFullyGated(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            in_size=[self.input_width,self.input_height],
            time_steps=self.time_steps
        ).to(self.device)

        self.assertTrue(
            self.conv_gru(x).size()
                ==
            torch.Size(
                (
                    self.batch_size,self.hidden_channels,self.input_width//stride[0],self.input_height//stride[1]
                )
            )
        )  

        kernel_size=[5,5]
        stride = [7,4]

        self.conv_gru = ConvGRUFullyGated(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            in_size=[self.input_width,self.input_height],
            time_steps=self.time_steps
        ).to(self.device)

        self.assertTrue(
            self.conv_gru(x).size()
                ==
            torch.Size(
                (
                    self.batch_size,self.hidden_channels,self.input_width//stride[0],self.input_height//stride[1]
                )
            )
        )    
  


    
    
class TestSampleConvGRUFullyGatedNet(unittest.TestCase):
    def setUp(self):
        print()
        self.in_channels = 1
        self.hidden_channels = 32
        self.kernel_size = [3,3]
        self.stride = [2,2]
        self.in_size = [28,28]

        self.device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
        self.model = SampleConvGRUFullyGatedNet(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
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

class TestSampleConvGRUType2Net(unittest.TestCase):
    def setUp(self):
        print()
        self.in_channels = 1
        self.hidden_channels = 32
        self.kernel_size = [3,3]
        self.stride = [2,2]
        self.in_size = [28,28]

        self.device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
        self.model = SampleConvGRUTye2Net(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            in_size=self.in_size
        ).to(self.device)
        torch.manual_seed(config.seed)


        train_kwargs = {'batch_size': config.batch_size}
        test_kwargs = {'batch_size': config.test_batch_size}
        if self.device.type=="cuda":
            cuda_kwargs = {'num_workers': 1,
                        'pin_memory': True,
                        'shuffle': True}
            train_kwargs.update(cuda_kwargs)
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