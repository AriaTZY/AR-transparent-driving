# 此py文件是专门供3d拼接和循视做准备的
import cv2 as cv
import numpy as np
import math
import time
import Feature.read as read
from camero.convolution_calibration import cal_RT, cal_FZ
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


# 由于python版本没有好用的copyTo函数，所以只能自己写一个，在传入的mask中，只用将需要叠加的部分弄白就行
# pos是左上角坐标，（u，v）制
def copyTo(src, sticker, pos, mask):
    if pos[0] <= 0:
        print('ROI裁剪部分超出图像负索引，放弃copy')
        return src
    else:
        if len(mask.shape) == 3:  # 如果输入的mask是三通道的
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        out = copy.copy(src)
        shape_logo = sticker.shape
        ROI = src[pos[1]:pos[1]+shape_logo[0], pos[0]:pos[0]+shape_logo[1]]  # 取出相应区域ROI
        # 首先做给ROI上画上黑色背景
        ret, mask_inv = cv.threshold(mask, 100, 255, cv.THRESH_BINARY_INV)
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


# 做两幅图像在不同俯仰角，偏航角下的全景拼接合并
def do():
    angel_part = -37
    yaw = 30
    pitch = 20
    # mid_l = read.get_pics(1, '/calibration/VR/mid_l')
    # left = read.get_pics(1, '/calibration/VR/left')
    mid_l = read.get_pics(1, '/calibration/VR/right')
    left = read.get_pics(1, '/calibration/VR/mid_r')
    while True:
        print('yaw=%d, pitch=%d' % (yaw, pitch))
        # 以下是计算公共矩阵部分
        view_map = []
        camera_map = Inverse(shot_angel=13.5, yaw_angel=0, height=1600, size=size)
        view_map.append(Inverse(shot_angel=pitch, yaw_angel=yaw, height=1600, size=size, y_offset=-100))
        view_map.append(Inverse(shot_angel=pitch, yaw_angel=yaw+angel_part, height=1600, size=size, y_offset=-100))
        # 开始变换
        trans_mid_l = get_perspective(camera_map, view_map[0])
        trans_left = get_perspective(camera_map, view_map[1])
        a_mid_l = cv.warpPerspective(mid_l, trans_mid_l, (size[1], size[0]))
        a_left = cv.warpPerspective(left, trans_left, (size[1], size[0]))
        # 手动拼接部分
        offset = caclulate_offset(camera_map, [trans_mid_l, trans_left], size, angel_part)
        joint = np.zeros([int(a_left.shape[0]+25), a_left.shape[1] + a_mid_l.shape[1] - offset[0], 3], np.uint8)
        A_position = [0, 20]  # 左图的左上角坐标位置，cv格式
        B_position = [a_left.shape[1], 20]  # 默认右图左上角坐标位置，cv格式
        B_position[0] = B_position[0] - offset[0]  # 对拼接图进行位移调整offset，如没有调整就是整幅拼接
        B_position[1] = B_position[1] - offset[1]
        A_shape = a_left.shape
        joint[A_position[1]:A_position[1]+A_shape[0], 0:A_shape[1]] = a_left
        # 以下是做mask中间图拼接工作
        ret, mask = cv.threshold(a_mid_l, 1, 255, cv.THRESH_BINARY)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = cv.dilate(mask, np.ones((3, 3), np.uint8))
        mask = cv.erode(mask, np.ones((3, 3), np.uint8), iterations=3)  # 使用膨胀后腐蚀操作可使mask消除接缝边缘效果更好
        joint = copyTo(joint, a_mid_l, B_position, mask)
        cv.imshow('a', a_mid_l)
        cv.imshow('b', a_left)
        cv.imshow('joint', joint)
        cv.setMouseCallback('joint', mouse_event)
        key = cv.waitKey(0)
        if key == 97:  # 左转
            yaw += 1
        elif key == 100:  # 右转
            yaw -= 1
        if key == 119:  # 向上抬
            pitch -= 1
        elif key == 115:  # 向下降
            pitch += 1


# 3幅图的拼接
def part_joint(left, middle, right, pitch=15, yaw=12):
    import time
    time_start = time.time()
    yaw_gap_left = -45
    yaw_gap_right = -37
    # 以下是计算公共矩阵部分
    view_map = []
    camera_map = Inverse(shot_angel=13.5, yaw_angel=0, height=1600, size=size)
    view_map.append(Inverse(shot_angel=pitch, yaw_angel=yaw, height=1600, size=size, y_offset=-100))
    view_map.append(Inverse(shot_angel=pitch, yaw_angel=yaw + yaw_gap_left, height=1600, size=size, y_offset=-100))
    view_map.append(Inverse(shot_angel=pitch, yaw_angel=yaw - yaw_gap_right, height=1600, size=size, y_offset=-100))
    # 开始变换
    trans_mid = get_perspective(camera_map, view_map[0])
    trans_left = get_perspective(camera_map, view_map[1])
    trans_right = get_perspective(camera_map, view_map[2])
    a_mid = cv.warpPerspective(middle, trans_mid, (size[1], size[0]))
    a_left = cv.warpPerspective(left, trans_left, (size[1], size[0]))
    a_right = cv.warpPerspective(right, trans_right, (size[1], size[0]))
    cv.imshow('left', a_left)
    cv.imshow('middle', a_mid)
    cv.imshow('right', a_right)
    # 手动拼接部分
    offset_1 = caclulate_offset(camera_map, [trans_mid, trans_left], size, yaw_gap_left)
    offset_2 = caclulate_offset(camera_map, [trans_right, trans_mid], size, yaw_gap_right)
    width = a_left.shape[1] + a_mid.shape[1] + a_right.shape[1] - offset_1[0] - offset_2[0]
    joint = np.zeros([int(a_left.shape[0]+25), width, 3], np.uint8)  # 创建新图
    # offset 坐标计算
    left_position = [0, 20]  # 左图的左上角坐标位置，cv格式
    middle_position = [a_left.shape[1], 20]  # 默认右图左上角坐标位置，cv格式
    middle_position[0] = middle_position[0] - offset_1[0]  # 对拼接图进行位移调整offset，如没有调整就是整幅拼接
    middle_position[1] = middle_position[1] - offset_1[1]
    right_position = copy.copy(middle_position)
    right_position[0] = right_position[0] + a_right.shape[1] - offset_2[0]
    right_position[1] = right_position[1] - offset_2[1]
    # 进行joint
    left_shape = a_left.shape
    joint[left_position[1]:left_position[1]+left_shape[0], 0:left_shape[1]] = a_left
    # 以下是做mask中间图拼接工作
    CHOICE = 1  # 选择中间块的拼接方式，1快速拼接，2融合拼接
    if CHOICE == 1:
        ret, mask = cv.threshold(a_mid, 1, 255, cv.THRESH_BINARY)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = cv.dilate(mask, np.ones((3, 3), np.uint8))
        mask = cv.erode(mask, np.ones((3, 3), np.uint8), iterations=3)  # 使用膨胀后腐蚀操作可使mask消除接缝边缘效果更好
        joint = copyTo(joint, a_mid, middle_position, mask)
    # 使用融合算法拼接中间图
    if CHOICE == 2:
        # 首先还是全部复制一遍
        new_left = copy.copy(joint)
        ret, mask = cv.threshold(a_mid, 1, 255, cv.THRESH_BINARY)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = cv.dilate(mask, np.ones((3, 3), np.uint8))
        mask = cv.erode(mask, np.ones((3, 3), np.uint8), iterations=3)  # 使用膨胀后腐蚀操作可使mask消除接缝边缘效果更好
        joint = copyTo(joint, a_mid, middle_position, mask)
        # 利用融合算法重写重叠区域
        top_left = np.dot(trans_mid, [0, 0, 1])  # 1-用中间图的左边缘做融合起点
        bottom_left = np.dot(trans_mid, [0, size[0], 1])
        start = int(min(top_left[0]/top_left[2], bottom_left[0]/bottom_left[2]))
        start = start + middle_position[0]  # 这是计算出的右图偏差
        new_mid = np.zeros([int(a_left.shape[0]+25), width, 3], np.uint8)  # 2-为了方便叠加，所以将中间图做进一个和拼接图一样大的图里
        new_mid[middle_position[1]:middle_position[1]+a_mid.shape[0], middle_position[0]:middle_position[0]+a_mid.shape[1]] = a_mid
        top_right = np.dot(trans_left, [size[1], 0, 1])  # 3-用左边图的右边缘做融合终点
        bottom_right = np.dot(trans_left, [size[1], size[0], 1])
        end = int(max(top_right[0] / top_right[2], bottom_right[0] / bottom_right[2]))
        process_width = end - start
        cv.imshow('new_mid', new_mid)
        for row in range(0, int(a_left.shape[0] + 25)):
            for col in range(start, end):
                alpha = (process_width - (col - start)) / process_width
                if new_mid[row][col][0] + new_mid[row][col][1] + new_mid[row][col][2] == 0:  # 如果右图黑色，就全覆盖左图
                    alpha = 1.
                if new_left[row][col][0] + new_left[row][col][1] + new_left[row][col][2] == 0:  # 如果左图黑色，就全覆盖右图
                    alpha = 0.
                joint[row][col] = new_left[row][col] * alpha + new_mid[row][col] * (1 - alpha)
    # mask右边图
    ret, mask = cv.threshold(a_right, 1, 255, cv.THRESH_BINARY)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    mask = cv.dilate(mask, np.ones((3, 3), np.uint8))
    mask = cv.erode(mask, np.ones((3, 3), np.uint8), iterations=3)  # 使用膨胀后腐蚀操作可使mask消除接缝边缘效果更好
    joint = copyTo(joint, a_right, right_position, mask)
    # cv.imwrite('joint4.jpg', joint)
    time_end = time.time()
    print('totally cost', time_end - time_start, 's')
    return joint


def global_joint():
    pic_1 = read.get_pics(1, '/calibration/VR/left')
    pic_2 = read.get_pics(1, '/calibration/VR/mid_l')
    pic_3 = read.get_pics(1, '/calibration/VR/mid_r')
    pic_4 = read.get_pics(1, '/calibration/VR/right')
    yaw = -51
    pitch = 19
    while True:
        print('yaw=%d, pitch=%d' % (yaw, pitch))
        if yaw > -21:
            pic = part_joint(pic_1, pic_2, pic_3, pitch, yaw)
        else:
            pic = part_joint(pic_2, pic_3, pic_4, pitch, yaw+37)
        cv.imshow('joint', pic)
        key = cv.waitKey(0)
        if key == 97:  # 左转
            yaw += 1
        elif key == 100:  # 右转
            yaw -= 1
        if key == 119:  # 向上抬
            pitch -= 1
        elif key == 115:  # 向下降
            pitch += 1


if __name__ == '__main__':
    global_joint()
    # do()
