import cv2 as cv
import copy
from VR_project.perspective_spport import *


# 3幅图的拼接
def part_joint(left, middle, right, set_left_yaw, set_right_yaw, pitch=10, yaw=0):
    yaw_gap_left = set_left_yaw
    yaw_gap_right = set_right_yaw
    fixed_h = cfg.camera_fixed_height
    fixed_p = cfg.camera_fixed_pitch
    wanted_h = fixed_h  # 目标观察高度暂时设置为一致
    # 以下是计算公共矩阵部分
    view_map = []
    camera_map = Inverse(shot_angel=fixed_p, yaw_angel=0, height=fixed_h, size=size)
    view_map.append(Inverse(shot_angel=pitch, yaw_angel=yaw, height=wanted_h, size=size, y_offset=-100))
    view_map.append(Inverse(shot_angel=pitch, yaw_angel=yaw + yaw_gap_left, height=wanted_h, size=size, y_offset=-100))
    view_map.append(Inverse(shot_angel=pitch, yaw_angel=yaw - yaw_gap_right, height=wanted_h, size=size, y_offset=-100))
    # 开始变换
    trans_mid = get_perspective(camera_map, view_map[0])
    trans_left = get_perspective(camera_map, view_map[1])
    trans_right = get_perspective(camera_map, view_map[2])
    a_mid = cv.warpPerspective(middle, trans_mid, (size[1], size[0]))
    a_left = cv.warpPerspective(left, trans_left, (size[1], size[0]))
    a_right = cv.warpPerspective(right, trans_right, (size[1], size[0]))
    # cv.imshow('left', a_left)
    # cv.imshow('middle', a_mid)
    # cv.imshow('right', a_right)
    # 手动拼接部分
    offset_1 = caclulate_offset(camera_map, [trans_mid, trans_left], size, yaw_gap_left)
    offset_2 = caclulate_offset(camera_map, [trans_right, trans_mid], size, yaw_gap_right)
    width = a_left.shape[1] + a_mid.shape[1] + a_right.shape[1] - offset_1[0] - offset_2[0]
    joint = np.zeros([int(a_left.shape[0]+25), width, 3], np.uint8)  # 创建新图
    # offset 坐标计算
    left_position = [0, 20]  # 左图的左上角坐标位置，cv格式，我们最好把合成图上面和下面都留一些黑边
    middle_position = [a_left.shape[1], 20]  # 默认右图左上角坐标位置，cv格式
    middle_position[0] = middle_position[0] - offset_1[0]  # 对拼接图进行位移调整offset，如没有调整就是整幅拼接
    middle_position[1] = middle_position[1] - offset_1[1]
    right_position = copy.copy(middle_position)
    right_position[0] = right_position[0] + a_right.shape[1] - offset_2[0]
    right_position[1] = right_position[1] - offset_2[1]
    right_position = pos_limiting(right_position, a_left.shape[0] + 25)  # 一定要做纵向高度限幅处理
    middle_position = pos_limiting(middle_position, a_left.shape[0] + 25)
    # 进行joint
    left_shape = a_left.shape
    joint[left_position[1]:left_position[1]+left_shape[0], 0:left_shape[1]] = a_left
    # mask右边图
    ret, mask = cv.threshold(a_right, 1, 255, cv.THRESH_BINARY)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    mask = cv.dilate(mask, np.ones((3, 3), np.uint8))
    mask = cv.erode(mask, np.ones((3, 3), np.uint8), iterations=3)  # 使用膨胀后腐蚀操作可使mask消除接缝边缘效果更好
    joint = copyTo(joint, a_right, right_position, mask)
    # 以下是做mask中间图拼接工作
    ret, mask = cv.threshold(a_mid, 1, 255, cv.THRESH_BINARY)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    mask = cv.dilate(mask, np.ones((3, 3), np.uint8))
    mask = cv.erode(mask, np.ones((3, 3), np.uint8), iterations=3)  # 使用膨胀后腐蚀操作可使mask消除接缝边缘效果更好
    joint = copyTo(joint, a_mid, middle_position, mask)
    return joint


# 基于特征点匹配方法的自动标定技术，在计算gap得时候，是mid-left（src-dst）
def pic_auto_cali(left_input, mid_input, right_input, winname):
    left = copy.copy(left_input)
    mid = copy.copy(mid_input)
    right = copy.copy(right_input)
    surf = cv.xfeatures2d.SURF_create()
    # 进行各自的特征点检测
    kp1, des1 = surf.detectAndCompute(mid, None)
    kp2, des2 = surf.detectAndCompute(left, None)
    # kp3, des3 = surf.detectAndCompute(right, None)
    # -----------------------进行左边部分的标定-------------------------- #
    # 匹配，具体匹配的参数描述详见 https://blog.csdn.net/weixin_44072651/article/details/89262277
    bf = cv.BFMatcher()
    good = []
    matches = bf.knnMatch(des1, des2, 2)
    init_thre = 0.4  # 初始化阈值
    while len(good) < 20:
        for m, n in matches:
            if m.distance < init_thre * n.distance:
                good.append([m])
        init_thre = init_thre + 0.05
    check_num = 20  # 需要进行匹配的点个数
    temp = []
    best = []
    for index, d in enumerate(good):  # 这里使用了遍历操作，index表示索引，d才是good中的match点
        temp.append(d[0].distance)  # 排列刚刚good中的欧氏距离，做进一步筛选
    temp_sort = copy.copy(temp)
    temp_sort.sort()
    for i in range(check_num):
        index = temp.index(temp_sort[i])
        best.append(good[index])  # 这里把good进一步到best
    # 进行绘制（为UI做样子，所以多停留几秒）
    merge = np.zeros([left.shape[0], left.shape[1]*2, 3], np.uint8)
    # 开始基于匹配点做放射变换，得到一个放射变化map
    src_point = []
    dst_point = []
    x_differ = []
    y_differ = []
    for i in range(check_num):
        src_index = best[i][0].queryIdx  # 用这个来获取原图中和目标图中的关键点索引
        dst_index = best[i][0].trainIdx
        src_point.append(kp1[src_index].pt)  # 这一步是把两个对应点的坐标取出来
        dst_point.append(kp2[dst_index].pt)
        x_differ.append(src_point[i][0]-dst_point[i][0])
        y_differ.append(src_point[i][1]-dst_point[i][1])
        # print('No.', i, ': x_gap ', src_point[i][0]-dst_point[i][0], ', y_gap ', src_point[i][1]-dst_point[i][1])
    x_mean_differ_left = int(np.mean(x_differ))
    y_mean_differ_left = int(np.mean(y_differ))
    # print(x_mean_differ_left, y_mean_differ_left)
    src_point = np.array(src_point, dtype=np.int)
    dst_point = np.array(dst_point, dtype=np.int)
    #########################################################################
    # 计算旋转角度，通过选取src和dst的对应两点之间的连线的角度值进行矫正(取10次连线)
    # 1、首先选取两点距离较长的两个点作为基准，有一个while判定函数，以下50设置为线长阈值
    # 2、用向量夹角公式计算出相应夹角
    theta = []  # cos theta 坐标格式
    for i in range(30):
        import random
        line_length = 0
        while line_length < 50:  # 索引过小会导致两点连线过近，于是需要限制
            a = random.randint(0, check_num-1)
            b = random.randint(0, check_num-1)
            # 这是测试线，要保证测试线的足够长才能保证计算的角度准确，否则继续while循环找距离较远的相邻两组
            test_line = [src_point[a][0] - src_point[b][0], src_point[a][1] - src_point[b][1]]
            line_length = math.sqrt(test_line[0] ** 2 + test_line[1] ** 2)  # 线长
            # print('线长为：', line_length)
        # 以上工作是取两个不同索引下标，以下工作是取得原图和目标图中的两个连线坐标
        src_line = [src_point[a][0] - src_point[b][0], src_point[a][1] - src_point[b][1]]  # 计算坐标形式下的直线
        dst_line = [dst_point[a][0] - dst_point[b][0], dst_point[a][1] - dst_point[b][1]]
        # 以下是计算内积（dot）和绝对值乘积用来运算cos theta值
        dot = src_line[0]*dst_line[0] + src_line[1]*dst_line[1]  # 内积公式 x1*x2+y1*y2
        #  绝对值乘积公式 |a|*|b|
        abs = math.sqrt(src_line[0] ** 2 + src_line[1] ** 2) * math.sqrt(dst_line[0] ** 2 + dst_line[1] ** 2)
        theta.append(math.acos(dot/abs)*180/math.pi)  # math自带的是弧度制
        # 计算角度
        # print(a, b, theta[i])
        # 画原图上的线
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        blue = random.randint(0, 255)
        cv.circle(mid, (src_point[a][0], src_point[a][1]), 5, (r, g, blue), -1)
        cv.circle(mid, (src_point[b][0], src_point[b][1]), 5, (r, g, blue), -1)
        cv.line(mid, (src_point[a][0], src_point[a][1]), (src_point[b][0], src_point[b][1]), (255, 255, 255), 1)
        # 画dst上的线
        cv.circle(left, (dst_point[a][0], dst_point[a][1]), 5, (r, g, blue), -1)
        cv.circle(left, (dst_point[b][0], dst_point[b][1]), 5, (r, g, blue), -1)
        cv.line(left, (dst_point[a][0], dst_point[a][1]), (dst_point[b][0], dst_point[b][1]), (255, 255, 255), 1)
        merge[:, 0:mid.shape[1]] = left
        merge[:, mid.shape[1]:mid.shape[1]*2] = mid
        cv.putText(merge, 'left calibration', (merge.shape[1] // 3, 30), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                   (255, 255, 0), 2)
        cv.imshow(winname, merge)
        cv.waitKey(20)
    # -----------------------进行右边部分的标定-------------------------- #
    # 匹配，具体匹配的参数描述详见 https://blog.csdn.net/weixin_44072651/article/details/89262277
    surf2 = cv.xfeatures2d.SURF_create()
    # 进行各自的特征点检测
    kp1, des1 = surf2.detectAndCompute(mid, None)
    kp2, des2 = surf2.detectAndCompute(right, None)
    bf2 = cv.BFMatcher()
    good = []
    matches = bf2.knnMatch(des1, des2, 2)
    init_thre = 0.4  # 初始化阈值
    while len(good) < 20:
        for m, n in matches:
            if m.distance < init_thre * n.distance:
                good.append([m])
        init_thre = init_thre + 0.05
    check_num = 20  # 需要进行匹配的点个数
    temp = []
    best = []
    for index, d in enumerate(good):  # 这里使用了遍历操作，index表示索引，d才是good中的match点
        temp.append(d[0].distance)  # 排列刚刚good中的欧氏距离，做进一步筛选
    temp_sort = copy.copy(temp)
    temp_sort.sort()
    for i in range(check_num):
        index = temp.index(temp_sort[i])
        best.append(good[index])  # 这里把good进一步到best
    # 进行绘制（为UI做样子，所以多停留几秒）
    merge = np.zeros([left.shape[0], left.shape[1] * 2, 3], np.uint8)
    # 开始基于匹配点做放射变换，得到一个放射变化map
    src_point = []
    dst_point = []
    x_differ = []
    y_differ = []
    for i in range(check_num):
        src_index = best[i][0].queryIdx  # 用这个来获取原图中和目标图中的关键点索引
        dst_index = best[i][0].trainIdx
        src_point.append(kp1[src_index].pt)  # 这一步是把两个对应点的坐标取出来
        dst_point.append(kp2[dst_index].pt)
        x_differ.append(src_point[i][0] - dst_point[i][0])
        y_differ.append(src_point[i][1] - dst_point[i][1])
    x_mean_differ_right = int(np.mean(x_differ))
    y_mean_differ_right = int(np.mean(y_differ))
    src_point = np.array(src_point, dtype=np.int)
    dst_point = np.array(dst_point, dtype=np.int)
    #########################################################################
    # 计算旋转角度，通过选取src和dst的对应两点之间的连线的角度值进行矫正(取10次连线)
    for i in range(30):
        import random
        line_length = 0
        while line_length < 50:  # 索引过小会导致两点连线过近，于是需要限制
            a = random.randint(0, check_num - 1)
            b = random.randint(0, check_num - 1)
            # 这是测试线，要保证测试线的足够长才能保证计算的角度准确，否则继续while循环找距离较远的相邻两组
            test_line = [src_point[a][0] - src_point[b][0], src_point[a][1] - src_point[b][1]]
            line_length = math.sqrt(test_line[0] ** 2 + test_line[1] ** 2)  # 线长
        # 画原图上的线
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        blue = random.randint(0, 255)
        cv.circle(mid, (src_point[a][0], src_point[a][1]), 5, (r, g, blue), -1)
        cv.circle(mid, (src_point[b][0], src_point[b][1]), 5, (r, g, blue), -1)
        cv.line(mid, (src_point[a][0], src_point[a][1]), (src_point[b][0], src_point[b][1]), (255, 255, 255), 1)
        # 画dst上的线
        cv.circle(right, (dst_point[a][0], dst_point[a][1]), 5, (r, g, blue), -1)
        cv.circle(right, (dst_point[b][0], dst_point[b][1]), 5, (r, g, blue), -1)
        cv.line(right, (dst_point[a][0], dst_point[a][1]), (dst_point[b][0], dst_point[b][1]), (255, 255, 255), 1)
        merge[:, 0:mid.shape[1]] = mid
        merge[:, mid.shape[1]:mid.shape[1] * 2] = right
        cv.putText(merge, 'right calibration', (merge.shape[1] // 3, 30), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                   (255, 255, 0), 2)
        cv.imshow(winname, merge)
        cv.waitKey(20)
    return x_mean_differ_left, y_mean_differ_left, -x_mean_differ_right, y_mean_differ_right, merge


# 进行图片上下平移，如果是负值说明需要将子图像（左、右）向上平移
def y_offset_pic(pic, bias):
    bias = int(bias)
    # 下面这些点的形式是x,y（注意！），这是做上下平移工作
    new_sub = np.zeros([pic.shape[0], pic.shape[1], 3], np.uint8)
    if bias < 0:
        new_sub[0:bias+pic.shape[0], :] = pic[-bias:pic.shape[0], :]  # 这是向上移动
    else:
        new_sub[bias:pic.shape[0], :] = pic[0:pic.shape[0]-bias, :]  # 向下移动
    return new_sub

