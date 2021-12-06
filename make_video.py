import numpy as np
import cv2
#读取一张图片
size = (784,784)

#完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
videowrite = cv2.VideoWriter(r'./out/video/spot_falling.mp4',fourcc,60,size)#60是帧数，size是图片尺寸
img_array=[]
for filename in [r'./out/images/spot_falling_{0}.png'.format(i) for i in range(900)]:
   img = cv2.imread(filename)
   if img is None:
    print(filename + " is error!")
    continue
   img_array.append(img)

for i in range(len(img_array)):
 videowrite.write(img_array[i])
print('end!')