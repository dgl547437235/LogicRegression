import torch 
import numpy as np
import torch.nn as nn
from Model import MobileV2
import os
from PIL import Image
import cv2 
from torchvision import transforms
x_transforms = transforms.Compose([
							transforms.Resize((256,256)),
							transforms.ToTensor(),
							transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
						])
import math
max_length=int(math.sqrt(256**2+256**2))
def main():
	net=MobileV2()#加载模型
	if(os.path.exists("net.pt")):
		net.load_state_dict(torch.load("net.pt"))
	img_path="img//119.jpg"
	img=Image.open(img_path)
	cv_img=cv2.imread(img_path)
	img=x_transforms(img).view(1,3,256,256)

	pred=net(img)
	length=int(pred[0][0]*max_length)
	x1,y1,x2,y2=int(pred[0][1]*256),int(pred[0][2]*256),int(pred[0][3]*256),int(pred[0][4]*256)
	cv2.circle(cv_img,(x1,y1),3,(0,255,255),-1)
	cv2.circle(cv_img,(x2,y2),3,(0,255,255),-1)
	#cv2.line(cv_img,(10,10),(10+length,10),(255,255,0),2)
	cv2.imshow("win",cv_img)
	cv2.waitKey(0)
	print(pred)


if __name__=="__main__":
	main()