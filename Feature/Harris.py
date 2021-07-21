import cv2 as cv
import numpy as np
from Feature import read
import copy


def Harris():
    name = 'Lena'
    pic = read.get_pics(0, name)
    print(pic.shape)
    # 求出x，y方向上的Ix，Iy矩阵图
    # 对于Sobel这里的trick，参照https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    # 一定要用浮点形式做运算，后再求绝对值，再转换回uint8
    picx = cv.Sobel(pic, cv.CV_64F, 1, 0)
    picy = cv.Sobel(pic, cv.CV_64F, 0, 1)
    picx = np.abs(picx)
    picy = np.abs(picy)
    picx = np.uint8(picx)
    picy = np.uint8(picy)
    # 得到IxIx, IyIy 和 IxIy的矩阵图
    picxx = np.zeros(picx.shape, dtype=np.float32)
    picyy = np.zeros(picx.shape, dtype=np.float32)
    picxy = np.zeros(picx.shape, dtype=np.float32)
    for i in range(picx.shape[0]):
        for j in range(picx.shape[1]):
            picxx[i][j] = picx[i][j]**2
            picyy[i][j] = picy[i][j]**2
            picxy[i][j] = float(picx[i][j]) * float(picy[i][j])
    print(picxy.dtype)
    # 高斯模糊滤波，我认为是最重要的一步
    picxx = cv.GaussianBlur(picxx, (3, 3), 5)
    picyy = cv.GaussianBlur(picyy, (3, 3), 5)
    picxy = cv.GaussianBlur(picxy, (3, 3), 5)
    # 求R = det(M)-0.04*trace(M)^2， trace表示对角线的和，也是两个特征值的和
    # 角点R是大正数，边缘是R大负数，平区域是|R|小
    R = np.zeros(picx.shape, dtype=np.float32)
    correspond = np.zeros(picx.shape, dtype=np.uint8)
    for i in range(picx.shape[0]):
        for j in range(picx.shape[1]):
            a = picxx[i][j]
            b = picyy[i][j]
            c = picxy[i][j]
            # 以下注释的是验证是否可以去掉det项，事实证明，如果不做高斯模糊项，
            # 是完全可以去掉的，但同时也就不会有角点检测出来了
            # v1 = (a*b - c*c)-0.04*(a+b)*(a+b)
            # v2 = - 0.04 * (a + b) * (a + b)
            # if v1 != v2:
            #     print('a', a, 'b', b, 'c', c)
            #     print('picx', picx[i][j], 'picy', picy[i][j])
            R[i][j] = (a*b - c*c)-0.04*(a+b)*(a+b)
            if R[i][j] > 1e+8:
                correspond[i][j] = 2  # 2表示角点
            elif R[i][j] < -1e+7:
                correspond[i][j] = 1  # 1表示边缘

    color_pic = read.get_pics(1, name)
    edge_color_pic = copy.copy(color_pic)
    for i in range(picx.shape[0]):
        for j in range(picx.shape[1]):
            if correspond[i][j] == 2:
                cv.circle(color_pic, (j, i), 1, (255, 255, 0))
            if correspond[i][j] == 1:
                cv.circle(edge_color_pic, (j, i), 1, (0, 0, 255))
    cv.imshow('coner', color_pic)
    cv.imshow('edge', edge_color_pic)

    # Shi-Tomasi Corner Detector
    # arg1:需要的角点数目
    # arg2:算法中lemda的最小阈值，两个lemda都大于它，就认为是角点，取值0-1
    # arg3:两个角点最小间隔
    # 若arg1给定数目小于利用lemda数值检测出的角点个数，第二个参数实际上是失效的
    corners = cv.goodFeaturesToTrack(pic, 100, 0.01, 10)
    corners = np.int16(corners)  # corners输出来是浮点型的
    shi_pic = read.get_pics(1, name)
    for i in corners:
        x, y = i.ravel()  # 因为很奇怪，corners的每个点输出格式为二维数组，即[[x, y]]
        cv.circle(shi_pic, (x, y), 3, (255, 0, 255), -1)
    cv.imshow('SHI', shi_pic)
    cv.waitKey(0)


if __name__ == '__main__':
    Harris()

