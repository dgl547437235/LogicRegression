from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
x_transforms = transforms.Compose([
							transforms.Resize((256,256)),
							transforms.ToTensor(),
							transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
						])
def get_samples():
	samples=[]
	with open("train.txt","r",encoding="utf-8-sig") as f:
		while(True):
			line=f.readline()
			if(line==""):
				break
			img_path=line.split(" ")[0]
			length=line.split(" ")[1]
			x1=line.split(" ")[2]
			y1=line.split(" ")[3]
			x2=line.split(" ")[4]
			y2=line.split(" ")[5].split("\n")[0]
			samples.append([img_path,length,x1,y1,x2,y2])
	return samples
#a=get_samples()
import math
max_length=int(math.sqrt(256**2+256**2))
class CaptchaData(Dataset):
    def __init__(self,):
        super(Dataset, self).__init__()
        self.Samples=get_samples()
    def __len__(self):

        return len(self.Samples)


    def __getitem__(self, index):
       
        imgPath=self.Samples[index][0]
        length=int(self.Samples[index][1])/256
        img=Image.open(imgPath)
        img_tensor=x_transforms(img)
        x1=int(self.Samples[index][2])/256
        y1=int(self.Samples[index][3])/256
        x2=int(self.Samples[index][4])/256
        y2=int(self.Samples[index][5])/256

        label=np.array([length,x1,y1,x2,y2]).astype(np.float32)


        return img_tensor,label

#cap=CaptchaData()
#dataload=DataLoader(cap,12,drop_last=False)
#for img,label in dataload:
#	print(img)
