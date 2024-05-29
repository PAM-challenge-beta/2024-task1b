# -*- coding: utf-8 -*-
"""
Created on Wed May 29 03:29:23 2024

@author: gabri
"""

#%% Import librairies
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


#%% Model Simple 
class simple_cnn(nn.Module):
    def __init__(self, output_dim, c=64):
        super(simple_cnn, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=c, kernel_size=8, stride=4, padding=0), nn.BatchNorm2d(c)) 
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(2*c))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(4*c))
        self.lin1 = nn.Linear(c*4, 128*output_dim)
        self.lin2 = nn.Linear(output_dim*128, output_dim*16)
        self.lin3 = nn.Linear(output_dim*16, output_dim)
        self.dout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x,_ = torch.max(x, 3, False)
        x,_ = torch.max(x, 2, True)
        x = x.view(x.size(0), -1) 
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x)) 
        x = self.dout(x)
        y = torch.sigmoid(self.lin3(x))
        return y
    
class simple_cnn2(nn.Module):
    def __init__(self, output_dim, c=64):
        super(simple_cnn2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=c, kernel_size=8, stride=4, padding=0), nn.BatchNorm2d(c)) 
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(2*c))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(4*c))
        self.lin1 = nn.Linear(c*4, 32*output_dim)
        self.lin2 = nn.Linear(output_dim*32, output_dim*16)
        self.lin3 = nn.Linear(output_dim*16, output_dim)
        self.dout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x,_ = torch.max(x, 3, False)
        x,_ = torch.max(x, 2, True)
        x = x.view(x.size(0), -1) 
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x)) 
        x = self.dout(x)
        y = torch.sigmoid(self.lin3(x))
        return y