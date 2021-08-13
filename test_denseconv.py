import unittest
import torch 
from denseconv import DenseLayer,DenseBlock,Transition,SampleDenseNet
from math import ceil
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import config
from utils import train, test

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

class TestSampleDenseNet(unittest.TestCase):
    def setUp(self):
        print()
        self.in_channels = 1
        self.num_layers = 6

        self.device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
        self.model = SampleDenseNet(
            in_channels=self.in_channels,
            num_layers=self.num_layers,

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