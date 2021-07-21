import cv2 as cv
import numpy as np
import math
import pylab
import Feature.read as read
import camero.convolution_calibration as inverse


size = (384, 512)
camera_angle = 18  # 本身摄像机自身俯仰角
camera_angle_arc = camera_angle/180.*math.pi
camera_height = 1600
camera_height_y = camera_height*math.cos(camera_angle_arc)
camera_height_z = camera_height*math.sin(camera_angle_arc)


# 之前我们在做从world到camera坐标是都是以camera为集，这样是很麻烦的，我们这次用world做基，后再求逆
def Inverse(shot_angel=30., height=1000, y_offset=0.):
    point_w = [0, -10, 0, 1]
    # 这里角度是负数是因为这时已经把世界坐标系作为基了，然后看正交坐标系下旋转的刚好是负方向
    RT = inverse.cal_RT(x_a=-(90 - shot_angel), y_a=0, z_a=0, x_t=0, y_t=y_offset, z_t=-height)
    RT = np.linalg.inv(RT)
    point_c = np.dot(RT, point_w)
    print("real_world:x:%d, y:%d, z:%d" % (point_w[0], point_w[1], point_w[2]))
    print("came_world:x:%d, y:%d, z:%d" % (point_c[0], point_c[1], point_c[2]))
    FZ = inverse.cal_FZ(f=35, w=size[1], h=size[0], dx=0.081)
    # 再进行相乘
    OUT = np.dot(FZ, RT)
    return OUT


# 得到放射变换矩阵，传入的有变换矩阵，pix->mm的变换关系，通过自动调整四个角的映射关系得到放射变换矩阵
def get_perspective_mtx(map, scale=1.):
    # 确定真实世界坐标的随机取几个点
    w_top_left = [-2000, -10000, 0, 1]
    w_top_right = [2000, -10000, 0, 1]
    w_bot_left = [-1000, -1000, 0, 1]
    w_bot_right = [1000, -1000, 0, 1]
    # 得到这些点在拍摄坐标系下的uv坐标
    map2 = Inverse(shot_angel=90, height=20000, y_offset=-10000)  # 把相机往前挪，看得更多
    try1 = np.dot(map2, w_top_left)
    try2 = np.dot(map2, w_top_right)
    try3 = np.dot(map2, w_bot_left)
    try4 = np.dot(map2, w_bot_right)
    try1 = try1 / try1[2]
    try2 = try2 / try2[2]
    try3 = try3 / try3[2]
    try4 = try4 / try4[2]
    n_p1 = [try1[0], try1[1]]
    n_p2 = [try2[0], try2[1]]
    n_p3 = [try3[0], try3[1]]
    n_p4 = [try4[0], try4[1]]
    print(n_p1, n_p2, n_p3, n_p4)
    point_dst = np.array([n_p1, n_p2, n_p3, n_p4], dtype=np.float32)
    # 确定映射后的图片上的uv值(透视问题中，属于dst)
    p1 = np.dot(map, w_top_left)
    p2 = np.dot(map, w_top_right)
    p3 = np.dot(map, w_bot_left)
    p4 = np.dot(map, w_bot_right)
    p1 = p1 / p1[2]
    p2 = p2 / p2[2]
    p3 = p3 / p3[2]
    p4 = p4 / p4[2]
    point_src = np.array([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]], [p4[0], p4[1]]], dtype=np.float32)
    print(point_src)
    TRANS_MTX = cv.getPerspectiveTransform(point_src, point_dst)
    return TRANS_MTX


def do():
    pic = read.get_pics(1, '/calibration/cali2')
    map = Inverse(shot_angel=18, height=1600)
    TRANS_MTX = get_perspective_mtx(map, scale=1.)
    show = cv.warpPerspective(pic, TRANS_MTX, (size[1], size[0]))
    cv.imshow('Origin', pic)
    cv.imshow('Inverse', show)
    cv.waitKey(0)


if __name__ == '__main__':
    do()
