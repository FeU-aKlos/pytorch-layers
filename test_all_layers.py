import unittest

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from denseconv import DenseBlock
from lstmconv import ConvLSTM

class TestAllLayers(unittest.TestCase):
    def setUp(self):
        # Training settings
        self.batch_size=64
        self.test_batch_size=1000
        self.epochs=3
        self.lr=1.0
        self.gamma=0.7
        self.no_cuda=False
        self.dry_run=False
        self.seed=1
        self.log_interval=10
        self.save_model=False

        self.use_cuda = not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        

        
        

        
    def test_main(self):

      

        

        
        ###
        #Conv
        ###
        print("Conv:\n")
        
        ###
        #RecurrentConv
        ###
        print("RecurrentConv:\n")
        model = RecurrentConvNet(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=self.kernel_size,stride=self.stride,in_size=self.in_size).to(self.device)

        optimizer = optim.Adadelta(model.parameters(), lr=self.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)
        for epoch in range(1, self.epochs + 1):
            _train(model, self.device, train_loader, optimizer, epoch)
            _test(model, self.device, test_loader)
            scheduler.step()
        
        ###
        #DenseNet
        ###
        print("DenseNet:\n")
        model = DenseNet(in_channels=self.in_channels).to(self.device)

        optimizer = optim.Adadelta(model.parameters(), lr=self.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)
        for epoch in range(1, self.epochs + 1):
            _train(model, self.device, train_loader, optimizer, epoch)
            _test(model, self.device, test_loader)
            scheduler.step()

        ###
        #ConvLSTM
        ###
        print("ConvLSTM:\n")
        model = ConvLSTMNet(in_channels=self.in_channels,hidden_channels=self.out_channels,kernel_size=self.kernel_size,stride=self.stride,in_size=self.in_size).to(self.device)

        optimizer = optim.Adadelta(model.parameters(), lr=self.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)
        for epoch in range(1, self.epochs + 1):
            _train(model, self.device, train_loader, optimizer, epoch)
            _test(model, self.device, test_loader)
            scheduler.step()

        

if __name__=="__main__":
    unittest.main()