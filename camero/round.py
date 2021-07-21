import cv2 as cv
import numpy as np
import math
import pylab
import Feature.read as read
import camero.convolution_calibration as inverse

# 图片尺寸大小
size = (512, 384)


# 在做inverse矩阵时，由于若以camera做基，和真实中的摄像机移动方向不正交，做位移处理时很麻烦
# 所以我们这里先以world做基，后再用求逆的方法进行转换运算
# x_angle是摄像机的俯仰角，z_angle是观察角度
def Inverse_3d(x_angle=30., z_angle=0., height=2000, radius=2000):
    # 计算旋转时的x，y位移
    z_angle_arc = z_angle/180. * math.pi
    x_trans = radius * math.sin(z_angle_arc)
    y_trans = radius * math.cos(z_angle_arc)
    point_w = [0, -10, 0, 1]
    # 首先以真实世界为基求 从摄像机坐标系->真实坐标系的变换关系
    RT = inverse.cal_RT(x_a=-(90-x_angle), y_a=0., z_a=z_angle, x_t=x_trans, y_t=y_trans, z_t=-height)
    RT = np.linalg.inv(RT)
    point_c = np.dot(RT, point_w)
    # print("real_world:x:%d, y:%d, z:%d" % (point_w[0], point_w[1], point_w[2]))
    # print("came_world:x:%d, y:%d, z:%d" % (point_c[0], point_c[1], point_c[2]))
    FZ = inverse.cal_FZ(f=35, w=size[1], h=size[0], dx=0.081)
    # 再进行相乘
    OUT = np.dot(FZ, RT)
    return OUT


# 得到放射变换矩阵，传入的有变换矩阵，pix->mm的变换关系，通过自动调整四个角的映射关系得到放射变换矩阵
def get_perspective_mtx(map, scale=20.):
    half_x = size[1]//2
    half_y = size[0]//2
    # 确定真实世界坐标(在透视问题中，这些点属于src)
    w_top_left = [-(half_x*scale), -(half_y*scale), 0, 1]
    w_top_right = [(half_x*scale), -(half_y*scale), 0, 1]
    w_bot_left = [-(half_x*scale), (half_y*scale), 0, 1]
    w_bot_right = [(half_x*scale), (half_y*scale), 0, 1]
    # 确定映射后的图片上的uv值(透视问题中，属于dst)
    p1 = np.dot(map, w_top_left)
    p2 = np.dot(map, w_top_right)
    p3 = np.dot(map, w_bot_left)
    p4 = np.dot(map, w_bot_right)
    p1 = p1 / p1[2]
    p2 = p2 / p2[2]
    p3 = p3 / p3[2]
    p4 = p4 / p4[2]
    point_dst = np.array([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]], [p4[0], p4[1]]], dtype=np.float32)
    point_src = np.array([[0, 0], [size[1], 0], [0, size[0]], [size[1], size[0]]], dtype=np.float32)
    TRANS_MTX = cv.getPerspectiveTransform(point_src, point_dst)
    return TRANS_MTX


def shot_from_3d():
    pic = read.get_pics(1, 'calibration/cali6')
    cv.imshow('Origin', pic)
    hei = 600
    x_angle_s = 40
    z_angle_s = 0
    while True:
        print('height=', hei)
        # 第一个值是图像俯视时图片高度对应的实际长度
        scale = 1500./size[0]
        map = Inverse_3d(x_angle=x_angle_s, z_angle=z_angle_s, height=hei, radius=600)
        TRANS_MTX = get_perspective_mtx(map, scale)
        show = cv.warpPerspective(pic, TRANS_MTX, (size[1], size[0]))
        cv.imshow('Inverse', show)
        cv.waitKey(100)
        z_angle_s = z_angle_s+10
        # hei = hei + 50
        # x_angle_s = x_angle_s+3


if __name__ == '__main__':
    shot_from_3d()
