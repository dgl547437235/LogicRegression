import cv2
import numpy as np
import time
from openvino.inference_engine import IECore
import math
sz=256
def main():
	ie=IECore()#初始化引擎
	net=ie.read_network("net.onnx")#读取网络
	input_blob = next(iter(net.input_info))#定义输入
	out_blob = next(iter(net.outputs))#定义输出
	cv_img=cv2.imread("img//963.jpg")#读取图片
	src = (cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)/ 255.0).astype(np.float32).transpose((2, 0, 1))#做一个图片转换
	exec_net = ie.load_network(network=net, device_name="CPU")#加载网络
	pred = exec_net.infer(inputs={input_blob: [src]})["output"]#执行推断
	x1,y1,x2,y2=int(pred[0][1]*sz),int(pred[0][2]*sz),int(pred[0][3]*sz),int(pred[0][4]*sz)#计算坐标
	length=int(pred[0][0]*int(math.sqrt(sz**2+sz**2)))#计算长度
	#显示
	cv2.circle(cv_img,(x1,y1),3,(0,255,255),-1)
	cv2.circle(cv_img,(x2,y2),3,(0,255,255),-1)
	cv2.putText(cv_img,"len:"+str(length),(10,20),1,1.0,(2,2,255),1)
	cv2.imshow("win",cv_img)
	cv2.waitKey(0)
if __name__=="__main__":
	main()