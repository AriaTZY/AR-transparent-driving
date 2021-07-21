import cv2 as cv
import CarDetection.ConvNet as conv
import random
import os.path
import numpy as np
import pylab


# 在图像中画出正方形，就可以自动保存
def callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print('x=%d, y=%d' % (x, y))


# 做滑窗是（40*60）的非极大值抑制
def non_maximum_suppression(input_, start_line=142, step=15):
    out = np.zeros(input_.shape, dtype=np.uint8)
    shaped = input_.shape
    for x in range(0, shaped[1] - 60, step):
        for y in range(start_line, shaped[0] - 40, step):
            center = (x+30, y+20)
            ROI = input_[y:y+40, x:x+60]
            max_val = np.max(ROI)
            if input_[center[1], center[0]] >= max_val and not max_val == 0:
                # out[center[1], center[0]] = 255
                out[center[1]-step:center[1]+step//2, center[0]-step:center[0]+step//2] = 255
    return out


# 添加热图
def add_heat(pic, axis_list, threshold):
    out = np.zeros(pic.shape, dtype=np.uint8)
    for i in range(len(axis_list)):
        x = axis_list[i][0]
        y = axis_list[i][1]
        out[y:y+40, x:x+60] += 25
    cv.imshow('heat_picture', out)
    out[out <= threshold] = 0
    return out


# 由于摄像机安装位置的原因，142往上都是景物，不用检测
def detection(step=15):
    class_conv = conv.LeNet_5((60, 40), Train_mode='Fast')
    for i in range(70, 139):
        print('No.pic', i)
        path = 'D:/show_pic/car_detection/'
        name = path + str(i) + '.jpg'
        show = cv.imread(name, 1)
        if show is None:
            print('打开出错')
            continue
        pic = cv.imread(name, 0)
        # pic = cv.resize(pic, (0, 0), None, 1.5, 1.5)
        # show = cv.resize(show, (0, 0), None, 1.5, 1.5)
        shaped = pic.shape
        count = 0
        for x in range(0, shaped[1]-60, step):
            for y in range(142, shaped[0]-40, step):
                count = count + 1
        print('total:', count)
        data_numpy = np.zeros([count, 60*40], dtype=np.float32)
        data_axis = np.zeros([count, 2], dtype=np.int)
        i = 0
        for x in range(0, shaped[1]-60, step):
            for y in range(142, shaped[0]-40, step):
                ROI = pic[y:y+40, x:x+60]
                data_numpy[i] = np.reshape(ROI, [1, 60*40])
                data_axis[i] = [x, y]
                i = i + 1
        class_conv.fast_data = data_numpy
        ret = class_conv.train_model()
        candidate_axis = []
        for i in range(len(ret)):
            if ret[i][0] == 1:
                x = data_axis[i][0]
                y = data_axis[i][1]
                candidate_axis.append([x, y])
                cv.circle(show, (x, y), 2, (255, 0, 100), -1)
                cv.rectangle(show, (x, y), (x + 60, y + 40), (244, 0, 255), 3)
        heat = add_heat(pic, candidate_axis, 25)
        non_max = non_maximum_suppression(heat, start_line=142, step=step)
        contours = cv.findContours(non_max, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        edge = np.asarray(contours[1])
        edge = np.reshape(edge, [edge.shape[0], -1, 2])
        show2 = cv.imread(name, 1)
        # edge中的形状是(2, ?, 2)，其中第一维表示总共有几个矩形
        for i in range(edge.shape[0]):
            print('绘制第', i, '个矩形')
            x = 0
            y = 0
            for j in range(edge.shape[1]):
                x += edge[i][j][0]
                y += edge[i][j][1]
            x = x//edge.shape[1]
            y = y//edge.shape[1]
            cv.circle(show, (x, y), 5, (0, 0, 255), -1)
            cv.rectangle(show2, (x-30, y-20), (x+30, y+20), (255, 100, 0), 2)
        # cv.drawContours(show, contours[1], -1, (0, 0, 255), 3)
        cv.imshow('non_max', non_max)
        cv.imshow('asas', heat)
        cv.imshow('car', show)
        cv.imshow('detection', show2)
        cv.setMouseCallback('car', callback)
        cv.waitKey(0)


# 与视频相结合的单张图片简单show
def single_detection(show, step=15):
    class_conv = conv.LeNet_5((60, 40), Train_mode='Fast')
    pic = cv.cvtColor(show, cv.COLOR_BGR2GRAY)
    shaped = pic.shape
    count = 0
    for x in range(0, shaped[1]-60, step):
        for y in range(142, shaped[0]-40, step):
            count = count + 1
    data_numpy = np.zeros([count, 60*40], dtype=np.float32)
    data_axis = np.zeros([count, 2], dtype=np.int)
    i = 0
    for x in range(0, shaped[1]-60, step):
        for y in range(142, shaped[0]-40, step):
            ROI = pic[y:y+40, x:x+60]
            data_numpy[i] = np.reshape(ROI, [1, 60*40])
            data_axis[i] = [x, y]
            i = i + 1
    class_conv.fast_data = data_numpy
    ret = class_conv.train_model()
    candidate_axis = []
    for i in range(len(ret)):
        if ret[i][0] == 1:
            x = data_axis[i][0]
            y = data_axis[i][1]
            candidate_axis.append([x, y])
    heat = add_heat(pic, candidate_axis, threshold=25)
    non_max = non_maximum_suppression(heat, start_line=142, step=step)
    cv.imshow('non_max', non_max)
    contours, hei, q = cv.findContours(non_max, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(show, contours[1], -1, (0, 0, 255), 3)
    cv.imshow('contours', show)
    print(type(contours))
    print(type(contours[0]))
    print(len(contours))
    print(contours)
    cv.waitKey(0)
    print('please check', contours[1])
    print(contours)
    edge = np.asarray(contours[1])
    # edge中的形状是(2, 4, 2)，其中第一维表示总共有几个矩形
    for i in range(edge.shape[0]):
        print('绘制第', i, '个矩形')
        x = 0
        y = 0
        for j in range(edge.shape[1]):
            x += edge[i][j][0][0]
            y += edge[i][j][0][1]
        x = x//edge.shape[1]
        y = y//edge.shape[1]
        cv.rectangle(show, (x-30, y-20), (x+30, y+20), (255, 0, 255), 3)
    return show


# 对视频进行保存为图片的函数
def dynamic_detection():
    cap = cv.VideoCapture('G:\\video\\DCIMA\\20190222_075212A.mp4')
    if not cap.isOpened():
        print('视频打开错误')
    count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            print('No.', count)
            frame = cv.resize(frame, (640, 360), None)
            if count >= 118:
                frame = single_detection(frame, step=15)
            cv.imshow('Frame', frame)
            cv.waitKey(1)
            count += 1
        else:
            break
    print('视频播放完毕')


if __name__ == '__main__':
    # dynamic_detection()
    detection()
