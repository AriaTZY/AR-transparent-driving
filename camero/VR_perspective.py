# 此py文件是专门供3d拼接和循视做准备的
import cv2 as cv
import numpy as np
import math
import pylab
import Feature.read as read
from camero.convolution_calibration import cal_RT, cal_FZ
from Joint.camera_joint import pic_joint
import copy

size = (384, 512)
camera_angle = 18  # 本身摄像机自身俯仰角
camera_angle_arc = camera_angle/180.*math.pi
camera_height = 1600
camera_height_y = camera_height*math.cos(camera_angle_arc)
camera_height_z = camera_height*math.sin(camera_angle_arc)

last_point = (0, 0)


def mouse_event(event,x,y,flags,param):
    global last_point
    if event == cv.EVENT_LBUTTONDOWN:
        last_point = (x, y)
        print('更新第一点', x, y)
    elif event == cv.EVENT_RBUTTONDOWN:
        print('off_set=', x-last_point[0], ',', y-last_point[1])


# 柔和拼接，rect的格式是（横0， 横1， 竖0， 竖1）
def soft_joint(screen, sticker, rect):
    sticker_x = 0
    for x in range(rect[0], rect[1]):
        sticker_y = 0
        for y in range(rect[2], rect[3]):
            origin = screen[y][x]
            if screen[y][x].any() == 0:
                origin = sticker[sticker_y][sticker_x]
            if sticker[sticker_y][sticker_x].any() == 0:
                sticker[sticker_y][sticker_x] = screen[y][x]
            add = sticker[sticker_y][sticker_x]
            origin = origin.astype(np.float)
            pixel = (add+origin)//2
            pixel = pixel.astype(np.uint8)
            # print('add=', add, 'origin=', origin, 'avr=', pixel)
            screen[y][x] = pixel
            sticker_y += 1
        sticker_x += 1


# 由于python版本没有好用的copyTo函数，所以只能自己写一个，在传入的mask中，只用将需要叠加的部分弄白就行
# pos是左上角坐标，（u，v）制
def copyTo(src, sticker, pos, mask):
    if len(mask.shape) == 3:  # 如果输入的mask是三通道的
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    out = copy.copy(src)
    shape_logo = sticker.shape
    ROI = src[pos[1]:pos[1]+shape_logo[0], pos[0]:pos[0]+shape_logo[1]]  # 取出相应区域ROI
    # 首先做给ROI上画上黑色背景
    ret, mask_inv = cv.threshold(mask, 100, 255, cv.THRESH_BINARY_INV)
    # cv.imshow('try', ROI)
    # cv.waitKey(0)
    mask_inv_BGR = cv.cvtColor(mask_inv, cv.COLOR_GRAY2BGR)
    joint_1 = cv.bitwise_and(ROI, mask_inv_BGR, mask_inv)
    # 接下来给转换成需要附加的彩色图片在黑色布景上
    mask_BGR = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    joint_2 = cv.bitwise_and(sticker, mask_BGR, mask)
    # 合并
    joint = cv.add(joint_1, joint_2)
    # joint = cv.addWeighted(joint_1, 0.5, joint_2, 0.9, 0.)
    # 绘制到大图中
    out[pos[1]:pos[1]+shape_logo[0], pos[0]:pos[0]+shape_logo[1]] = joint
    return out



# 使用掩码拼接
def mask_joint():
    logo = read.get_pics(1, 'logo')
    b = read.get_pics(1)
    ROI = b[0:logo.shape[0], 0:logo.shape[1]]
    logo_gray = cv.cvtColor(logo, cv.COLOR_BGR2GRAY)
    # 首先我们做正向阈值计算
    ret, logo_mask = cv.threshold(logo_gray, 220, 255, cv.THRESH_BINARY)
    logo_mask_color = cv.cvtColor(logo_mask, cv.COLOR_GRAY2BGR)  # 和上面那个一样，只不过这个狗日的要我保持三通道
    # 反向阈值计算
    ret, logo_mask_inv = cv.threshold(logo_gray, 220, 255, cv.THRESH_BINARY_INV)
    logo_mask_inv_color = cv.cvtColor(logo_mask_inv, cv.COLOR_GRAY2BGR)  # 和上面那个一样，只不过这个狗日的要我保持三通道

    joint1 = cv.bitwise_and(ROI, logo_mask_color, mask=logo_mask)  # 这个函数是对掩码上白色部分做处理，黑色部分保留黑色
    joint2 = cv.bitwise_and(logo, logo_mask_inv_color, mask=logo_mask_inv)
    cv.imshow('joint1', joint1)
    cv.imshow('joint2', joint2)
    out = cv.add(joint1, joint2)
    cv.imshow('out', out)
    cv.waitKey(0)


# 之前我们在做从world到camera坐标是都是以camera为集，这样是很麻烦的，我们这次用world做基，后再求逆
def Inverse(shot_angel=30., yaw_angel=0., height=1000, size=(512, 384), y_offset=0.):
    # 必须进行这样一种计算，因为基是摄像头坐标系，真实坐标在基中并不是仅限于x轴的平移
    point_w = [0, -10, 0, 1]
    # 这里角度是负数是因为这时已经把世界坐标系作为基了，然后看正交坐标系下旋转的刚好是负方向
    RT = cal_RT(x_a=-(90 - shot_angel), y_a=0, z_a=yaw_angel, x_t=0, y_t=y_offset, z_t=-height)
    RT = np.linalg.inv(RT)
    FZ = cal_FZ(f=35, w=size[1], h=size[0], dx=0.081)
    # 再进行相乘
    OUT = np.dot(FZ, RT)
    return OUT


# 输入世界坐标，得到uv坐标，多传入的分别是俯仰角和偏航角，偏航角左偏为正
def get_uv_point(pitch_a=30., yaw_a=0, height=None, size=(384, 512)):
    map = Inverse(shot_angel=13.5, yaw_angel=0, height=1600, size=size)
    # 确定真实世界坐标
    w_top_left = [-1000, -20000, 0, 1]
    w_top_right = [1000, -20000, 0, 1]
    w_bot_left = [-1000, -1000, 0, 1]
    w_bot_right = [1000, -1000, 0, 1]
    # 另外一种方法处理，通过设定俯视高度得到不同视野
    input_height = height
    if height is None:
        input_height = 1600
    map2 = Inverse(shot_angel=pitch_a, yaw_angel=yaw_a, height=input_height, size=(512, 384), y_offset=-100)  # 把相机往前挪，看得更多
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
    point_dst = np.array([n_p1, n_p2, n_p3, n_p4], dtype=np.float32)
    # 确定映射后的图片上的uv值
    p1 = np.dot(map, w_top_left)
    p2 = np.dot(map, w_top_right)
    p3 = np.dot(map, w_bot_left)
    p4 = np.dot(map, w_bot_right)
    p1 = p1/p1[2]
    p2 = p2/p2[2]
    p3 = p3/p3[2]
    p4 = p4/p4[2]
    point_src = np.array([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]], [p4[0], p4[1]]], dtype=np.float32)
    TRANS_MTX = cv.getPerspectiveTransform(point_src, point_dst)
    return TRANS_MTX


# 做同一点在两张图片下的位置标注，第二、三个参数是图1，图2的宽度
def caclulate_offset(yaw, width_left, width_mid):
    angel_part = 45  # 两相机安放角度差
    mid_l = read.get_pics(1, '/calibration/VR/mid_l')
    left = read.get_pics(1, '/calibration/VR/left')
    # 取得图像的透视变化矩阵
    trans_mid_l = get_uv_point(pitch_a=20, yaw_a=yaw, height=1600, size=size)
    trans_left = get_uv_point(pitch_a=20, yaw_a=yaw - angel_part, height=1600, size=size)
    # 取得在不同拍摄角度下的世界坐标的初步转换关系
    W_point_mid = [-3000, -10000, 0, 1]
    quarter_trans = cal_RT(x_a=0, y_a=0, z_a=-45, x_t=0, y_t=0, z_t=0)
    W_point_left = np.dot(quarter_trans, W_point_mid)
    # 取得两个机位的world-uv坐标的变换，由于拍摄机位的参数一样，两个相机公用一个变换
    primary_trans = Inverse(shot_angel=13.5, yaw_angel=0, height=1600, size=size)

    # 开始变换middle下的点uv坐标
    camera_uv_mid = np.dot(primary_trans, W_point_mid)
    camera_uv_mid = camera_uv_mid/camera_uv_mid[2]
    camera_uv_mid = camera_uv_mid[:3]
    new_uv_mid = np.dot(trans_mid_l, camera_uv_mid)
    new_uv_mid = new_uv_mid/new_uv_mid[2]
    new_uv_mid = new_uv_mid.astype(np.int)
    # 开始变换left下的点uv坐标
    camera_uv_left = np.dot(primary_trans, W_point_left)
    camera_uv_left = camera_uv_left / camera_uv_left[2]
    camera_uv_left = camera_uv_left[:3]
    new_uv_left = np.dot(trans_left, camera_uv_left)
    new_uv_left = new_uv_left / new_uv_left[2]
    new_uv_left = new_uv_left.astype(np.int)
    # 标出点位置
    pic_mid = cv.warpPerspective(mid_l, trans_mid_l, (size[1], size[0]))
    pic_left = cv.warpPerspective(left, trans_left, (size[1], size[0]))
    # 求offset
    x_offset = width_left - new_uv_left[0]
    x_offset = x_offset + new_uv_mid[0] - (size[1] - width_mid)
    y_offset = new_uv_mid[1] - new_uv_left[1]
    cv.circle(pic_mid, (new_uv_mid[0], new_uv_mid[1]), 3, (255, 0, 100), 3)
    cv.circle(pic_left, (new_uv_left[0], new_uv_left[1]), 3, (0, 100, 250), 3)
    cv.imshow('adad', pic_mid)
    cv.imshow('adad2', pic_left)
    # cv.waitKey(0)
    return x_offset, y_offset


# 做两幅图像在不同俯仰角，偏航角下的全景拼接合并
def do():
    angel_part = 45
    yaw = 30
    for yaw in range(20, 100, 2):
        print(yaw, 'degree level')
        mid_l = read.get_pics(1, '/calibration/VR/mid_l')
        left = read.get_pics(1, '/calibration/VR/left')
        trans_mid_l = get_uv_point(pitch_a=40, yaw_a=yaw, height=1600, size=size)
        trans_left = get_uv_point(pitch_a=40, yaw_a=yaw-angel_part, height=1600, size=size)
        a_mid_l = cv.warpPerspective(mid_l, trans_mid_l, (size[1], size[0]))
        a_left = cv.warpPerspective(left, trans_left, (size[1], size[0]))
        # 做裁剪计算部分
        middle_edge = []
        left_edge = []
        edge_point = [(0, 0, 1), (0, size[0], 1), (size[1], 0, 1), (size[1], size[0], 1)]  # 取左下角和左上角
        for i in range(len(edge_point)):
            temp = np.dot(trans_mid_l, edge_point[i])
            temp = temp/temp[2]
            middle_edge.append(temp)
            temp = np.dot(trans_left, edge_point[i])
            temp = temp / temp[2]
            left_edge.append(temp)
        middle_border = min(int(middle_edge[0][0]), int(middle_edge[1][0]))
        left_border = max(left_edge[2][0], left_edge[3][0])
        middle_border = int(max(0, middle_border))
        left_border = int(max(0, left_border))
        # 做裁剪部分
        a_mid_l = a_mid_l[:, middle_border:size[1], :]
        a_left = a_left[:, 0:left_border, :]
        cv.imshow('left', a_left)
        cv.imshow('middle', a_mid_l)
        # 手动拼接部分
        joint = np.zeros([int(a_left.shape[0]*1.2), a_left.shape[1] + a_mid_l.shape[1], 3], np.uint8)
        offset = caclulate_offset(20, a_left.shape[1], a_mid_l.shape[1])
        A_position = [0, 20]  # 左图的左上角坐标位置，cv格式
        B_position = [a_left.shape[1], 20]  # 默认右图左上角坐标位置，cv格式
        B_position[0] = B_position[0] - offset[0]  # 对拼接图进行位移调整offset，如没有调整就是整幅拼接
        B_position[1] = B_position[1] - offset[1]
        A_shape = a_left.shape
        B_shape = a_mid_l.shape
        joint[A_position[1]:A_position[1]+A_shape[0], 0:A_shape[1]] = a_left
        test = copy.copy(joint)
        # test[B_position[1]:B_position[1]+B_shape[0], B_position[0]:B_position[0]+B_shape[1]] = a_mid_l
        # soft_joint(test, a_mid_l, [B_position[0], B_position[0]+B_shape[1], B_position[1], B_position[1]+B_shape[0]])
        # cv.imshow('way1', test)
        ret, mask = cv.threshold(a_mid_l, 1, 255, cv.THRESH_BINARY)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = cv.dilate(mask, np.ones((3, 3), np.uint8))
        mask = cv.erode(mask, np.ones((3, 3), np.uint8), iterations=3)  # 使用膨胀后腐蚀操作可使mask消除接缝边缘效果更好
        joint = copyTo(joint, a_mid_l, B_position, mask)
        cv.imshow('joint', joint)
        cv.setMouseCallback('joint', mouse_event)
        # out = pic_joint(a_mid_l, a_left)
        # cv.imshow('joint', out)
        cv.waitKey(10)


if __name__ == '__main__':
    do()
