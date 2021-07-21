import cv2 as cv
import numpy as np
import math
import pylab
import Feature.read as read


# 在相机内参数的标定中，标定矩阵为K，乘法规则是(u, v, 1) = K*(Xc, Yc, Zc)
# [f/dx, 0,    u0]
# [0,    f/dy, v0]
# [0,    0,     1]
# 其中dx表示每个元像素点表示实际坐标下的长度，dx=36.0(cm)/1080(pix)

# 在相机的外参数中，首先我们看看旋转矩阵，若沿着z轴，则矩阵R元素应用于x，y两个向量
# R=[cos0 , sin0]
#   [-sin0, cos0]
# 我们在这里，只演示在z轴方向上的旋转参数构造方法，R在实际上是3*3的矩阵，因为还要保证不变的那一维度为 1
# [Xw, Yw, Zw, 1] = [Xc, Yc, Zc, 1]*[Rz, 0]
#                                   [0 , 1]
# 接下来我们再关注平移矩阵：
# T=[tx, ty, tz]
# 最后R=Rx*Ry*Rz，最终矩阵是
# [R, 0]
# [T, 1]
# 当然， 以上写的内容都是基于从Xc为中心，但这是不好操作的，实际应当是一个线性的传递过程
# 转换关系为：世界3d---（外参）--->相机3d---（内参）--->平面


# 创建w和c关系的矩阵，方向c=R*w，角度输入为角度制
def cal_RT(x_a=90., y_a=0., z_a=0., x_t=3., y_t=0., z_t=0.):
    # 创建单体矩阵
    Rx = np.zeros([3, 3], dtype=np.float)
    Ry = np.zeros([3, 3], dtype=np.float)
    Rz = np.zeros([3, 3], dtype=np.float)
    # 先转换为弧度制
    x_a = x_a / 180. * math.pi
    y_a = y_a / 180. * math.pi
    z_a = z_a / 180. * math.pi
    # 计算绕x旋转矩阵
    Rx[1][1] = math.cos(x_a)
    Rx[1][2] = math.sin(x_a)
    Rx[2][1] = -math.sin(x_a)
    Rx[2][2] = math.cos(x_a)
    Rx[0][0] = 1.
    # 计算绕y旋转矩阵
    Ry[0][0] = math.cos(y_a)
    Ry[0][2] = math.sin(y_a)
    Ry[2][0] = -math.sin(y_a)
    Ry[2][2] = math.cos(y_a)
    Ry[1][1] = 1.
    # 计算绕z旋转矩阵
    Rz[0][0] = math.cos(z_a)
    Rz[0][1] = math.sin(z_a)
    Rz[1][0] = -math.sin(z_a)
    Rz[1][1] = math.cos(z_a)
    Rz[2][2] = 1.
    # 计算混合R旋转矩阵
    R = np.dot(Rx, Ry)
    R = np.dot(Rz, R)

    # 创建总矩阵
    RT = np.zeros([4, 4], dtype=np.float)
    RT[:3, :3] = R
    RT[0][3] = x_t
    RT[1][3] = y_t
    RT[2][3] = z_t
    RT[3][3] = 1
    # print('Rotation+Transform:\n', RT)
    return RT


# 这是从相机坐标系到图像坐标系，f是焦距，和真实世界尺度统一用mm
# 应用于[u, v, 1] = A*[Xc, Yc, Zc, 1].T
# dx表示 1pix = dx(mm)，在手动标定中，dx=像平面的mm距离/像素个数
def cal_FZ(f=30, w=100, h=100, dx=200., dy=None):
    u0 = w//2
    v0 = h//2
    F = np.zeros([3, 4], dtype=np.float)
    Z = np.zeros([4, 3], dtype=np.float)  # 为了其次求逆
    # 创建从相机坐标到相平面的关系（此时得到的相平面上的尺度衡量仍然为cm）
    F[0][0] = f
    F[1][1] = f
    F[2][2] = 1.
    # 创建从物理度量单位到像素度量的关系
    if dy is None:
        dy = dx
    Z[0][0] = 1./dx
    Z[1][1] = 1./dy
    Z[2][2] = 1.
    Z[0][2] = u0
    Z[1][2] = v0
    FZ = np.dot(Z, F)
    FZ[3][3] = 1
    return FZ


# 从世界坐标系转换为像素坐标系，Pw是一个元组，剩下的是两个外参
def transform(Pw, R, T):
    Pw_np = np.zeros([4, 1], dtype=float)
    Pw_np[0][0] = Pw[0]
    Pw_np[1][0] = Pw[1]
    Pw_np[2][0] = Pw[2]
    Pw_np[3][0] = 1.
    RT = cal_RT(R[0], R[1], R[2], T[0], T[1], T[2])
    FZ = cal_FZ()
    A = np.dot(FZ, RT)
    out = np.dot(A, Pw_np)
    print(out)
    return out


# # 进行重映射，返回重映射矩阵
# def Inverse(shot_angel=30, height=1000, size=(512, 384)):
#     # 必须进行这样一种计算，因为基是摄像头坐标系，真实坐标在基中并不是仅限于x轴的平移
#     shot_angel_arc = shot_angel/180.*math.pi
#     z_trans = height*math.sin(shot_angel_arc)
#     y_trans = height*math.cos(shot_angel_arc)
#     point_w = [0, -10, 0, 1]
#     RT = cal_RT(x_a=90 - shot_angel, y_a=0, z_a=0, x_t=0, y_t=y_trans, z_t=z_trans)
#     point_c = np.dot(RT, point_w)
#     print("real_world:x:%d, y:%d, z:%d" % (point_w[0], point_w[1], point_w[2]))
#     print("came_world:x:%d, y:%d, z:%d" % (point_c[0], point_c[1], point_c[2]))
#     FZ = cal_FZ(f=35, w=size[1], h=size[0], dx=0.081)
#     # 再进行相乘
#     OUT = np.dot(FZ, RT)
#     return OUT


# 之前我们在做从world到camera坐标是都是以camera为集，这样是很麻烦的，我们这次用world做基，后再求逆
def Inverse(shot_angel=30., height=1000, size=(512, 384), y_offset=0.):
    # 必须进行这样一种计算，因为基是摄像头坐标系，真实坐标在基中并不是仅限于x轴的平移
    shot_angel_arc = shot_angel/180.*math.pi
    z_trans = height*math.sin(shot_angel_arc)
    y_trans = height*math.cos(shot_angel_arc)
    point_w = [0, -10, 0, 1]
    # 这里角度是负数是因为这时已经把世界坐标系作为基了，然后看正交坐标系下旋转的刚好是负方向
    RT = cal_RT(x_a=-(90 - shot_angel), y_a=0, z_a=0, x_t=0, y_t=y_offset, z_t=-height)
    RT = np.linalg.inv(RT)
    point_c = np.dot(RT, point_w)
    print("real_world:x:%d, y:%d, z:%d" % (point_w[0], point_w[1], point_w[2]))
    print("came_world:x:%d, y:%d, z:%d" % (point_c[0], point_c[1], point_c[2]))
    FZ = cal_FZ(f=35, w=size[1], h=size[0], dx=0.081)
    # 再进行相乘
    OUT = np.dot(FZ, RT)
    return OUT


# 输入世界坐标，得到uv坐标
def get_uv_point(size=(384, 512), max_length=20000, min_length=1500):
    pic = read.get_pics(1, '/calibration/cali2')
    map = Inverse(shot_angel=17.5, height=1600, size=size)
    # map = Inverse(shot_angel=40, height=1600, size=size)
    mm2pix = size[0]/(max_length-min_length)  # 用最大距离限制求出俯视图中真实和像素之间的关系
    # 确定真实世界坐标
    # w_top_left = [-1000, -max_length, 0, 1]
    # w_top_right = [1000, -max_length, 0, 1]
    # w_bot_left = [-1000, -min_length, 0, 1]
    # w_bot_right = [1000, -min_length, 0, 1]
    w_top_left = [-1000, -max_length, 0, 1]
    w_top_right = [1000, -max_length, 0, 1]
    w_bot_left = [-1000, -min_length, 0, 1]
    w_bot_right = [1000, -min_length, 0, 1]
    print('世界坐标系:')
    print(w_top_left, w_top_right, w_bot_left, w_bot_right)
    # 确定上述四个坐标在新图中的像素位置，使用尽头距离来获得真实坐标与俯视uv的关系
    n_p1 = [w_top_left[0]*mm2pix + size[1]//2, size[0] + w_top_left[1]*mm2pix]
    n_p2 = [w_top_right[0] * mm2pix + size[1]//2, size[0] + w_top_right[1] * mm2pix]
    n_p3 = [w_bot_left[0] * mm2pix + size[1]//2, size[0]]
    n_p4 = [w_bot_right[0] * mm2pix + size[1]//2, size[0]]
    # 另外一种方法处理，通过设定俯视高度得到不同视野
    map2 = Inverse(shot_angel=90, height=20000, size=(512, 384), y_offset=-8000)  # 把相机往前挪，看得更多
    try1 = np.dot(map2, w_top_left)
    try2 = np.dot(map2, w_top_right)
    try3 = np.dot(map2, w_bot_left)
    try4 = np.dot(map2, w_bot_right)
    print('人眼坐标系:')
    print(try1, try2, try3, try4)
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
    # 确定映射后的图片上的uv值
    p1 = np.dot(map, w_top_left)
    p2 = np.dot(map, w_top_right)
    p3 = np.dot(map, w_bot_left)
    p4 = np.dot(map, w_bot_right)
    print('相机坐标系:')
    print(p1, p2, p3, p4)
    p1 = p1/p1[2]
    p2 = p2/p2[2]
    p3 = p3/p3[2]
    p4 = p4/p4[2]
    point_show = np.array([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]], [p4[0], p4[1]]], dtype=np.int)
    point_src = np.array([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]], [p4[0], p4[1]]], dtype=np.float32)
    TRANS_MTX = cv.getPerspectiveTransform(point_src, point_dst)
    print(point_show[0], point_show[1], point_show[2], point_show[3])
    # 画到pic上
    cv.circle(pic, (point_show[0][0], point_show[0][1]), 5, (255, 0, 255), 2)
    cv.circle(pic, (point_show[1][0], point_show[1][1]), 5, (255, 0, 255), 2)
    cv.circle(pic, (point_show[2][0], point_show[2][1]), 5, (255, 255, 0), 2)
    cv.circle(pic, (point_show[3][0], point_show[3][1]), 5, (255, 255, 0), 2)
    cv.imshow('mark', pic)
    show = cv.warpPerspective(pic, TRANS_MTX, (size[1], size[0]))
    cv.imshow('Inverse', show)
    # cv.imwrite('origin.jpg', pic)
    # cv.imwrite('after.jpg', show)
    cv.waitKey(0)
    print(TRANS_MTX)


def loop_connect(size=(384, 512), max_length=10000):
    angle = 17
    while True:
        pic = read.get_pics(1, '/calibration/cali2')
        map = Inverse(shot_angel=angle, height=1600, size=size)
        distance = max_length
        angle = angle + 3
        print(angle)
        while True:
            # 确定真实世界坐标
            w_top_left = [-1000, -distance, 0, 1]
            w_top_right = [1000, -distance, 0, 1]
            # 确定映射后的图片上的uv值
            p1 = np.dot(map, w_top_left)
            p2 = np.dot(map, w_top_right)
            p1 = p1 / p1[2]
            p2 = p2 / p2[2]
            # 画到pic上
            cv.circle(pic, (int(p1[0]), int(p1[1])), 5, (255, 0, 255), 2)
            cv.circle(pic, (int(p2[0]), int(p2[1])), 5, (255, 0, 255), 2)
            cv.imshow('mark', pic)
            cv.waitKey(20)
            distance = distance - 100
            if distance < 0:
                break


if __name__ == '__main__':
    # loop_connect()
    get_uv_point()
    Inverse(shot_angel=30, height=300)
