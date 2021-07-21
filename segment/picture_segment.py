# 使用分水岭算法进行图像分割
import cv2 as cv
import numpy as np
import Feature.read as read
import pylab
from camero.camero_calibration import cali_live

# # 这是图像处理课的作业
# name = 'D:/show_pic/wallpaper/6f1fcbe4b208a963acf2f1e26b5545f6.jpg'
# ima = cv.imread(name, 0)
# ima = cv.resize(ima, (0, 0), None, 0.4, 0.4)
# bit_ima = np.zeros([8, ima.shape[0], ima.shape[1]], np.uint8)
# for i in range(ima.shape[0]):  # 列遍历
#     for j in range(ima.shape[1]):  # 行遍历
#         temp = ima[i][j]
#         bin_temp = bin(temp)
#         # print(bin_temp)
#         for layer_index in range(1, 9):  # python中最后索引是从-1开始的，所以这里不写成0开始
#             if bin_temp[-layer_index] is 'b':
#                 break
#             else:
#                 bit_ima[layer_index-1][i][j] = bin_temp[-layer_index]
# print('All DONE!')
# # 用pylab进行显示
# for index in range(8):
#     title_name = str(index) + ' bit'
#     pylab.subplot(2, 4, index+1)
#     pylab.imshow(bit_ima[index], 'gray')
#     pylab.axis('off')
#     pylab.title(title_name)
# pylab.show()
# cv.imshow('hhh', ima)
# print(bit_ima[0].shape)
# cv.waitKey(0)
#
# fps = 3
# size = (320, 320)
# video = cv.VideoWriter("VideoTest3.avi", cv.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
# for i in range(998):
#     name = 'C:/Users/11868/Desktop/-/tzy_faces(4-fully-connected-batchnorm)/epoch' + str(i) + '.jpg'
#     ima = cv.imread(name, 1)
#     test = 'epoch = ' + str(i)
#     cv.putText(ima, test, (10, 30), cv.FONT_HERSHEY_PLAIN, 2, (200, 200, 0), 3)
#     # cv.imshow('add', ima)
#     # cv.waitKey(0)
#     video.write(ima)
# print('视频输出完成！')

cap2 = cv.VideoCapture(1)
cap3 = cv.VideoCapture(2)
while True:
    sucess2, img2 = cap2.read()
    sucess3, img3 = cap3.read()
    img2 = img2[60:420, :]  # opencv的读入格式长宽比有些问题，所以重新裁剪
    img3 = img3[60:420, :]
    img2 = cali_live(img2)
    img3 = cali_live(img3)
    cv.imshow('window2', img2)
    cv.imshow('window3', img3)
    cv.waitKey(1)


# pic = read.get_pics(0, '/calibration/coin')
# ret, thre_pic = cv.threshold(pic, 190, 255, cv.THRESH_BINARY)
# thre_pic = cv.dilate(thre_pic, np.ones((3, 3), np.uint8), iterations=3)
# distance_pic = cv.distanceTransform(thre_pic, cv.DIST_L2, 3)
# cv.normalize(distance_pic, distance_pic, 0., 1., cv.NORM_MINMAX)
# cv.imshow('pic', pic)
# cv.imshow('aaa', thre_pic)
# cv.imshow('dis', distance_pic)
# cv.waitKey(0)

