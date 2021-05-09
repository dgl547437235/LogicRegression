from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=stride, padding=1, groups=planes,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1,stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileV2(nn.Module):
	def __init__(self):
		super(MobileV2,self).__init__()


		self.backbone=nn.Sequential(
			nn.Conv2d(3,16,3,1,1),
			nn.ReLU(True),
			nn.MaxPool2d(2,2),
			Block(16,32,3,1),
			Block(32,32,3,2),			
			Block(32,32,5,1),			
			Block(32,32,5,2),
			Block(32,64,5,2),
			Block(64,64,3,1),
			Block(64,32,3,2),
			Block(32,32,3,1),
			Block(32,32,3,2),
			Block(32,16,3,1),
			#nn.AvgPool2d(4)
			)
		self.fc=nn.Sequential(
			nn.Linear(16*4*4,32),
			nn.Linear(32,5),
	)
	def forward(self,x):
		y=self.backbone(x)
		out=self.fc(y.view(-1,16*4*4))
		return out



	
	





