import cv2 as cv
import numpy as np
from Feature import read
import pylab
import copy


# 计算某点的Hessian矩阵处理
def Hessian(pic, x=2, y=2, type='Dxx'):
    if type == 'Dxx':
        out = -2*pic[y][x] + pic[y][x-1] + pic[y][x+1]
    elif type == 'Dyy':
        out = -2*pic[y][x] + pic[y-1][x] + pic[y+1][x]
    elif type == 'Dxy':
        out = pic[y-1][x-1] + pic[y+1][x+1] - pic[y-1][x+1] + pic[y+1][x-1]
    return out


def max_hold(pic, diff_1, diff_2, diff_3, window=5, width=512, times=1):
    for x in np.arange(0, width-window, 1):
        for y in np.arange(0, width-window, 1):
            temp = np.zeros([3, window, window])
            center = diff_2[y+window//2][x+window//2]
            temp[0] = diff_1[y:y+window, x:x+window]
            temp[1] = diff_2[y:y + window, x:x + window]
            temp[2] = diff_3[y:y + window, x:x + window]
            temp[1][window//2][window//2] = temp[1][0][0]
            if center > np.max(temp) or center < np.min(temp):
                pic[y+window//2*times][x+window//2*times] = 255
    return pic


def Self_SIFT(Octave_num=3, Scale_num=3, init_sigma=1.6):
    name = 'Lena'
    pic = read.get_pics(0, name)
    pic_color = read.get_pics(1, name)
    width = pic.shape[1]
    height = pic.shape[0]
    k = pow(2, 1/3.)  # sigma增益
    # 创建Octave矩阵，1-Octave维度索引，2-Scale索引，3,4-长宽
    # 1. 构建高斯金字塔
    # first Octave s=3 real_layer=s+3=6
    Guassion = np.zeros([Octave_num, Scale_num + 3, height, width], np.uint8)
    oct_pic = pic
    for index_oct in range(Octave_num):
        for index_scl in range(Scale_num+3):
            sigma = init_sigma*(k**index_scl)
            oct_wide = oct_pic.shape[1]
            oct_height = oct_pic.shape[0]
            print('Guassion:size:%d, Octave:%d, Scale:%d, sigma:%f' % (oct_pic.shape[1], index_oct, index_scl, sigma))
            Guassion[index_oct][index_scl][0:oct_height, 0:oct_wide] = cv.GaussianBlur(oct_pic, (5, 5), sigma)
            # 绘制
            pylab.subplot(Octave_num, Scale_num+3, index_oct*(Scale_num+3)+index_scl+1)
            pylab.imshow(Guassion[index_oct][index_scl], 'gray')
            pylab.axis('off')
        # 在一次Octave做完之后，下采样
        temp = Guassion[index_oct][Scale_num-2]  # 取-2的那张图是因为正好是2倍的高斯模糊核sigma，符合原论文
        oct_pic = cv.resize(temp, (0, 0), oct_pic, 0.5, 0.5)
    pylab.show()

    # 2. 构建DOG金字塔
    DOG = np.zeros([Octave_num, Scale_num + 2, height, width], np.uint8)
    for index_oct in range(Octave_num):
        for index_scl in np.arange(1, Scale_num + 3, 1):
            upper = np.asarray(Guassion[index_oct][index_scl], np.float32)
            lower = np.asarray(Guassion[index_oct][index_scl - 1], np.float32)
            temp = np.abs(upper - lower)
            temp = temp.astype(dtype=np.uint8)
            # cv.threshold(temp, 1, 255, cv.THRESH_BINARY, dst=temp)
            cv.normalize(temp, temp, 255, 0, cv.NORM_MINMAX)
            DOG[index_oct][index_scl-1] = temp
            print('DOG:Octave:%d, Scale:%d' % (index_oct, index_scl))
            # 绘制
            pylab.subplot(Octave_num, Scale_num + 2, index_oct * (Scale_num + 2) + index_scl)
            pylab.imshow(DOG[index_oct][index_scl-1], 'gray')
            pylab.axis('off')
    pylab.show()

    # 3. 做非极大值抑制，并且此处整合所有DOG层的key点至一张图上
    check_point_pic = np.zeros(pic.shape, np.uint8)
    for index_oct in range(Octave_num):
        time = index_oct + 1
        wide = pic.shape[0]//(index_oct+1)
        for index_scl in np.arange(1, Scale_num + 1, 1):
            print('Check_1:Octave:%d, Scale:%d' % (index_oct, index_scl))
            check_point_pic = max_hold(check_point_pic, DOG[index_oct][index_scl-1], DOG[index_oct][index_scl],
                                       DOG[index_oct][index_scl+1], window=3, width=wide, times=time)
    # 绘制
    pylab.imshow(check_point_pic, 'gray')
    pylab.axis('off')
    pylab.show()

    # 4. 做Hessian矩阵去除边沿影响识别
    check_point_pic_1 = np.zeros(pic.shape, np.uint8)
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            if check_point_pic[i][j] == 255:
                Dxx = Hessian(pic, j, i, 'Dxx')
                Dyy = Hessian(pic, j, i, 'Dyy')
                Dxy = Hessian(pic, j, i, 'Dxy')
                Tr_H = Dxx + Dyy
                Det_H = Dxx * Dyy - Dxy * Dxy
                delta = Tr_H**2/float(Det_H)  # 判别式
                print('x,y=%d,%d, delta=%f' % (j, i, delta))
                if delta > 10:
                    check_point_pic_1[i][j] = 255
    # 绘制
    pylab.imshow(check_point_pic_1, 'gray')
    pylab.axis('off')
    pylab.show()


def SIFT():
    name_1 = 'SIFT2'
    pic_1 = read.get_pics(0, name_1)
    color_pic_1 = read.get_pics(1, name_1)
    name_2 = 'SIFT1'
    pic_2 = read.get_pics(0, name_2)
    color_pic_2 = read.get_pics(1, name_2)
    sift = cv.xfeatures2d.SIFT_create()
    # detectAndCompute和detect的区别在于，前者还会返回一个128维度的向量特征，用于匹配，后者只返回点信息
    kp1, des1 = sift.detectAndCompute(pic_1, None)
    kp2, des2 = sift.detectAndCompute(pic_2, None)
    # 进行关键点匹配
    bf = cv.BFMatcher()
    good = []
    matches = bf.knnMatch(des1, des2, 2)
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            print(m.distance, ',', n.distance)
            good.append([m])
    show = cv.drawMatchesKnn(pic_1, kp1, pic_2, kp2, good, None)
    cv.imshow('good', show)
    # 取前10个距离最短的点进行匹配
    check_num = 40  # 需要进行匹配的点个数
    temp = []
    best = []
    for index, d in enumerate(good):
        temp.append(d[0].distance)
    temp_sort = copy.copy(temp)
    temp_sort.sort()
    for i in range(check_num):
        index = temp.index(temp_sort[i])
        print('index=', index, 'val=', temp_sort[i])
        best.append(good[index])
    show = cv.drawMatchesKnn(pic_1, kp1, pic_2, kp2, best, None)
    cv.imshow('best', show)
    cv.waitKey(0)


def SURF():
    name_1 = 'Lena'
    pic_1 = read.get_pics(0, name_1)
    color_pic_1 = read.get_pics(1, name_1)
    name_2 = 'SIFT1'
    pic_2 = read.get_pics(0, name_2)
    color_pic_2 = read.get_pics(1, name_2)

    # 全套SURF检测
    surf = cv.xfeatures2d.SURF_create()
    kp = surf.detect(pic_1, None)
    show = cv.drawKeypoints(color_pic_1, kp, None)
    cv.imshow('show', show)

    # 对于不需要检测方向的情况，我们打开Upright选项
    surf.setUpright(True)
    kp = surf.detect(pic_1, None)
    show = cv.drawKeypoints(color_pic_1, kp, None)
    cv.imshow('SURF_Upright', show)

    # 带匹配的
    surf.setUpright(False)
    kp, des = surf.detectAndCompute(pic_1, None)
    show = cv.drawKeypoints(color_pic_1, kp, None)
    cv.imshow('Compute', show)

    cv.waitKey(0)


if __name__ == '__main__':
    # Self_SIFT()
    # SIFT()
    SURF()
