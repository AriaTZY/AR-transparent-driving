# 此py文件是专门供3d拼接和循视做准备的
import cv2 as cv
import numpy as np
import math
import serial
import read_support as read
from matrix_support import cal_RT, cal_FZ, cali_live
import copy
import config as cfg

size = cfg.picture_size
camera_angle = 18  # 本身摄像机自身俯仰角
camera_angle_arc = camera_angle/180.*math.pi
camera_height = 1600
camera_height_y = camera_height*math.cos(camera_angle_arc)
camera_height_z = camera_height*math.sin(camera_angle_arc)

last_point = (0, 0)


def mouse_event(event, x, y, flags, param):
    global last_point
    if event == cv.EVENT_LBUTTONDOWN:
        last_point = (x, y)
        print('更新第一点', x, y)
    elif event == cv.EVENT_RBUTTONDOWN:
        print('off_set=', x-last_point[0], ',', y-last_point[1])


# 由于python版本没有好用的copyTo函数，所以只能自己写一个，在传入的mask中，只用将需要叠加的部分弄白就行
# pos是左上角坐标，（u，v）制
def copyTo(src, sticker, pos, mask):
    # copyTo(joint, a_right, right_position, mask)
    if pos[0] <= 0:
        print('ROI裁剪部分超出图像负索引，放弃copy')
        return src
    else:
        if len(mask.shape) == 3:  # 如果输入的mask是三通道的
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        out = copy.copy(src)
        shape_logo = sticker.shape
        # print('position', pos)
        ROI = src[pos[1]:pos[1]+shape_logo[0], pos[0]:pos[0]+shape_logo[1]]  # 取出相应区域ROI
        # print('height', pos[1], ':', pos[1]+shape_logo[0], 'width', pos[0], ':', pos[0]+shape_logo[1])
        # 首先做给ROI上画上黑色背景
        ret, mask_inv = cv.threshold(mask, 100, 255, cv.THRESH_BINARY_INV)
        mask_inv_BGR = cv.cvtColor(mask_inv, cv.COLOR_GRAY2BGR)
        # print('原图', src.shape, 'mask', mask.shape)
        # print('ROI', ROI.shape)
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
    # 这里角度是负数是因为这时已经把世界坐标系作为基了，然后看正交坐标系下旋转的刚好是负方向
    RT = cal_RT(x_a=-(90 - shot_angel), y_a=0, z_a=yaw_angel, x_t=0, y_t=y_offset, z_t=-height)
    RT = np.linalg.inv(RT)
    FZ = cal_FZ(f=35, w=size[1], h=size[0], dx=0.081)
    # 再进行相乘
    OUT = np.dot(FZ, RT)
    return OUT


# 得到二维图像的透视变换关系，只用传入摄像机的map矩阵和观察角度的map矩阵
def get_perspective(camera_map, view_map):
    map = camera_map
    # 确定真实世界坐标
    w_top_left = [-1000, -20000, 0, 1]
    w_top_right = [1000, -20000, 0, 1]
    w_bot_left = [-1000, -1000, 0, 1]
    w_bot_right = [1000, -1000, 0, 1]
    map2 = view_map
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


# 做同一点在两张图片下的位置标注，第二、三个参数是图1，图2的宽度，输入第二个参数是角度差
def caclulate_offset(camera_map, perspective_map_list, size, angel_part=-45):
    # 取得图像的透视变化矩阵
    trans_mid_l = perspective_map_list[0]
    trans_left = perspective_map_list[1]
    # 取得在不同拍摄角度下的世界坐标的初步转换关系
    W_point_mid = [-3000, -10000, 0, 1]
    quarter_trans = cal_RT(x_a=0, y_a=0, z_a=angel_part, x_t=0, y_t=0, z_t=0)
    W_point_left = np.dot(quarter_trans, W_point_mid)
    # 取得两个机位的world-uv坐标的变换，由于拍摄机位的参数一样，两个相机公用一个变换
    primary_trans = camera_map
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
    # 求offset
    x_offset = size[1] - new_uv_left[0]
    x_offset = x_offset + new_uv_mid[0]
    y_offset = new_uv_mid[1] - new_uv_left[1]
    return x_offset, y_offset


# 为了防止在向一侧过度倾斜时，纵向pos(角点位置)过于靠下使得裁剪图片过小，导致知名错误的问题
# 横向宽度不存在这个问题，因为在创建joint图片的时候就是依照他们的宽度总和来做的
# 注意pos[0]代表宽度， pos[1]代表高度
def pos_limiting(pos, pic_height):
    max_pos_1 = pic_height - size[0]
    out_pos = copy.copy(pos)
    if out_pos[1] > max_pos_1:
        out_pos[1] = max_pos_1
    return out_pos


def global_joint(mode=1):
    if mode == 1:
        # pic1 = cv.imread('left.jpg')
        # pic2 = pic3 = cv.imread('middle.jpg')
        pic2 = cv.imread('../img_lib/base.jpg')
        pic1 = pic3 = cv.imread('../img_lib/left_2.jpg')
        pic1 = read.crop(pic1)
        pic2 = read.crop(pic2)
        pic3 = read.crop(pic3)
    else:
        cap1 = cv.VideoCapture(1)
        cap2 = cv.VideoCapture(3)
        cap3 = cv.VideoCapture(0)
    yaw = cfg.look_init_yaw
    pitch = cfg.look_init_pitch
    while True:
        if mode == 2:
            sucess1, pic1 = cap1.read()
            sucess2, pic2 = cap2.read()
            sucess3, pic3 = cap3.read()
            pic1 = pic1[60:420, :]  # opencv的读入格式长宽比有些问题，所以重新裁剪
            pic2 = pic2[60:420, :]
            pic3 = pic3[60:420, :]
            pic1 = cali_live(pic1)
            pic2 = cali_live(pic2)
            pic3 = cali_live(pic3)
        print('yaw=%d, pitch=%d' % (yaw, pitch))
        if yaw < cfg.max_look_right:  # 限幅处理
            yaw = cfg.max_look_right
        elif yaw > cfg.max_look_left:  # 限幅处理
            yaw = cfg.max_look_left
        pic = part_joint(pic3, pic2, pic1, pitch, yaw)
        cv.imshow('joint', pic)
        key = cv.waitKey(1)
        if key == 97:  # 左转
            yaw += 5
        elif key == 100:  # 右转
            yaw -= 5
        if key == 119:  # 向上抬
            pitch -= 2
        elif key == 115:  # 向下降
            pitch += 3


def serial_control(left_yaw, left_height):
    port_list = list(serial.tools.list_ports.comports())
    if len(port_list) == 0:
        print('串口出错，找不到串口！！')
    else:
        for i in range(0, len(port_list)):
            print(port_list[i])
            str_port = str(port_list[i])
            if str_port.find('CH340') != -1:  # 如果找到了串口，并且串口是带有CH340
                print(str_port[0:4])
                com_name = str_port[0:4]
    ser = serial.Serial(com_name, 115200)
    last_yaw = 0
    last_picth = 0
    # 1,2,3 分别是左中右顺序 这个原则不能变
    camera_order = cfg.camera_order
    cap1 = cv.VideoCapture(camera_order[0])
    cap2 = cv.VideoCapture(camera_order[1])
    pitch = cfg.look_init_pitch
    serial_count = 20  # 这个是用来卡读取串口的间隔时间的，若读取过于频繁会造成图像卡顿
    while True:
        # 处理串口
        if serial_count >= 1:
            ser.write('\n'.encode())
            recv = ser.readline()
            recv = str(recv)
            print(recv)
            try:
                if recv[2] == 'y':
                    yaw = int(recv[3:6])
                    pitch = int(recv[7:10])
                else:
                    yaw = last_yaw
                    pitch = last_picth
            except Exception:
                yaw = last_yaw
                pitch = last_picth
            yaw = yaw - 360
            pitch = pitch - 360
            print('yaw=', yaw, 'pitch=', pitch)
            last_yaw = yaw
            last_picth = pitch
            serial_count = 0
        # 处理图像
        sucess1, pic1 = cap1.read()
        sucess2, pic2 = cap2.read()
        pic1 = pic1[60:420, :]  # opencv的读入格式长宽比有些问题，所以重新裁剪
        pic2 = pic2[60:420, :]
        # 这个是进行160度畸变的处理
        pic1 = cali_live(pic1)
        pic2 = cali_live(pic2)
        # 进行上下移动处理
        from UI_support import y_offset_pic
        pic1 = y_offset_pic(pic1, left_height)
        if yaw < cfg.max_look_right:  # 限幅处理
            yaw = cfg.max_look_right
        elif yaw > cfg.max_look_left:  # 限幅处理
            yaw = cfg.max_look_left
        from UI_support import part_joint
        pic = part_joint(pic1, pic2, left_yaw, pitch, yaw)
        pic = cv.resize(pic, (0, 0), None, 1.65, 1.5)
        cv.imshow('Window', pic)
        serial_count += 1
        a = cv.waitKey(1)
        if a == 98:  # 如果点击退出，则退出
            cv.destroyWindow('Window')
            break


# 这一部分是使用摇杆
def serial_control_rocker(left_yaw, left_height):
    port_list = list(serial.tools.list_ports.comports())
    if len(port_list) == 0:
        print('串口出错，找不到串口！！')
    else:
        for i in range(0, len(port_list)):
            print(port_list[i])
            str_port = str(port_list[i])
            if str_port.find('CH340') != -1:  # 如果找到了串口，并且串口是带有CH340
                print(str_port[0:4])
                com_name = str_port[0:4]
    ser = serial.Serial(com_name, 115200)
    yaw_float = 0.
    pitch_float = 0.
    # 1,2,3 分别是左中右顺序 这个原则不能变
    camera_order = cfg.camera_order
    cap1 = cv.VideoCapture(camera_order[0])
    cap2 = cv.VideoCapture(camera_order[1])
    pitch = cfg.look_init_pitch
    serial_count = 20  # 这个是用来卡读取串口的间隔时间的，若读取过于频繁会造成图像卡顿
    speed = 3.5  # 旋转速度
    while True:
        # 处理串口
        if serial_count >= 1:
            ser.write('\n'.encode())
            recv = ser.readline()
            recv = str(recv)
            print(recv)
            try:
                rev_yaw = int(recv[3:6])  # 反映在轴上是x轴方向
                rev_pitch = int(recv[7:10])  # 反映在轴上是y轴方向
            except Exception:
                rev_yaw = rev_pitch = 0
            rev_yaw = rev_yaw - 500
            rev_pitch = rev_pitch - 500
            print(rev_yaw, rev_pitch)
            if rev_yaw > 300:
                yaw_float -= speed
            elif rev_yaw < -300:
                yaw_float += speed
            if rev_pitch > 300:
                pitch_float += speed
            elif rev_pitch < -300:
                pitch_float -= speed
            # 限幅前的数据
            yaw = int(yaw_float)
            pitch = int(pitch_float)
            if yaw_float > 80:
                yaw = yaw_float = 80
            if yaw_float < -80:
                yaw = yaw_float = -80
            if pitch_float > 40:
                pitch = pitch_float = 40
            if pitch_float < -40:
                pitch = pitch_float = -40
            serial_count = 0
        # 处理图像
        sucess1, pic1 = cap1.read()
        sucess2, pic2 = cap2.read()
        pic1 = pic1[60:420, :]  # opencv的读入格式长宽比有些问题，所以重新裁剪
        pic2 = pic2[60:420, :]
        # 这个是进行160度畸变的处理
        pic1 = cali_live(pic1)
        pic2 = cali_live(pic2)
        # 进行上下移动处理
        from UI_support import y_offset_pic
        pic1 = y_offset_pic(pic1, left_height)
        if yaw < cfg.max_look_right:  # 限幅处理
            yaw = cfg.max_look_right
        elif yaw > cfg.max_look_left:  # 限幅处理
            yaw = cfg.max_look_left
        from UI_support import part_joint
        pic = part_joint(pic1, pic2, left_yaw, pitch, yaw)
        pic = cv.resize(pic, (0, 0), None, 1.65, 1.5)
        cv.imshow('Window', pic)
        serial_count += 1
        a = cv.waitKey(1)
        if a == 98:  # 如果点击退出，则退出
            cv.destroyWindow('Window')
            break


def cali_camera_idx():
    cap1 = cv.VideoCapture(0)
    cap2 = cv.VideoCapture(1)
    cap3 = cv.VideoCapture(2)
    cap4 = cv.VideoCapture(3)
    while 1:
        sucess1, pic1 = cap2.read()
        sucess2, pic2 = cap1.read()
        sucess3, pic3 = cap3.read()
        sucess4, pic4 = cap4.read()
        cv.imshow('img0', pic1)
        cv.imshow('img1', pic2)
        cv.imshow('img2', pic3)
        cv.imshow('img3', pic4)
        cv.waitKey(1)


if __name__ == '__main__':
    # cali_camera_idx()
    global_joint(mode=1)
    # serial_control()
