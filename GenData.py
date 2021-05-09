import cv2
import numpy as np
import random
import math
import os
def get_random_color():
	return (random.randint(0,255),random.randint(0,255),random.randint(0,255))

def GenData():
	if(not os.path.exists("img")):
		os.mkdir("img")
	file=open("train.txt","w",encoding="utf-8-sig")
	for i in range(1000):
		zero_mat=np.zeros([256,256,3],np.uint8)
		loc_x1=random.randint(0,256)
		loc_y1=random.randint(0,256)
		loc_x2=random.randint(0,256)
		loc_y2=random.randint(0,256)
		length=int(math.sqrt((loc_x1-loc_x2)**2+(loc_y1-loc_y2)**2))
		cv2.line(zero_mat,(loc_x1,loc_y1),(loc_x2,loc_y2),get_random_color(),2)

		img_path=os.path.join("img",str(i)+".jpg")
		cv2.imwrite(img_path,zero_mat)
		file.write(img_path+" "+str(length)+" "+str(loc_x1)+" "+str(loc_y1)+" "+str(loc_x2)+" "+str(loc_y2)+"\n")
	file.close()

if __name__=="__main__":
	GenData()
