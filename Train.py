import torch 
import numpy as np
import torch.nn as nn
from Model import MobileV2
from Dataset import CaptchaData
from torch.utils.data import DataLoader,Dataset
import os

def main():
	net=MobileV2()#加载模型
	if(os.path.exists("net.pt")):
		print("load weight")
		net.load_state_dict(torch.load("net.pt"))
	net=net.cuda()
	data_cap=CaptchaData()#加载数据集
	data_loader=DataLoader(data_cap,30,True)#加载数据通道
	optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)#定义优化器
	criterion=torch.nn.SmoothL1Loss()
	criterion=torch.nn.MSELoss()
	for epoch in range(10000):
		net.train()
		losses=[]
		for i,(img,label) in enumerate(data_loader):
			img,label=img.cuda(),label.cuda()
			pred=net(img)
			loss=criterion(pred,label)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			losses.append(loss.item())
		if(epoch%5==0):
			print("loss:%5f"%(np.mean(np.array(losses))))
			torch.save(net.state_dict(),"net.pt")
if __name__=="__main__":
	main()