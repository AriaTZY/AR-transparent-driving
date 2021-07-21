import Feature.read as read
import cv2 as cv
import numpy as np
import copy
import math


# 这是一个全面的测试函数，不提供向外部调用
def pic_joint_test():
    # base = read.get_pics(1, '/calibration/joint/base')
    # sub = read.get_pics(1, '/calibration/joint/catch2')
    base = cv.imread('../img_lib/base.jpg')
    sub = cv.imread('../img_lib/left_1.jpg')
    base = read.crop(base)
    sub = read.crop(sub)
    shaped = sub.shape
    surf = cv.xfeatures2d.SURF_create()
    # 进行各自的特征点检测
    kp1, des1 = surf.detectAndCompute(base, None)
    kp2, des2 = surf.detectAndCompute(sub, None)
    # 匹配，具体匹配的参数描述详见 https://blog.csdn.net/weixin_44072651/article/details/89262277
    bf = cv.BFMatcher()
    good = []
    matches = bf.knnMatch(des1, des2, 2)
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good.append([m])
    show_good = cv.drawMatchesKnn(base, kp1, sub, kp2, good, None)
    # cv.imshow('hhh', show_good)
    # cv.waitKey(0)
    # 取前10个最佳匹配点
    check_num = 10  # 需要进行匹配的点个数
    temp = []
    best = []
    for index, d in enumerate(good):  # 这里使用了遍历操作，index表示索引，d才是good中的match点
        temp.append(d[0].distance)  # 排列刚刚good中的欧氏距离，做进一步筛选
    temp_sort = copy.copy(temp)
    temp_sort.sort()
    for i in range(check_num):
        index = temp.index(temp_sort[i])
        best.append(good[index])  # 这里把good进一步到best
    show_best = cv.drawMatchesKnn(base, kp1, sub, kp2, best, None)  # 再进行绘制
    cv.imshow('best', show_best)
    # 开始基于匹配点做放射变换，得到一个放射变化map
    src_point = []
    dst_point = []
    for i in range(10):
        src_index = best[i][0].queryIdx  # 用这个来获取原图中和目标图中的关键点索引
        dst_index = best[i][0].trainIdx
        src_point.append(kp1[src_index].pt)  # 这一步是把两个对应点的坐标取出来
        dst_point.append(kp2[dst_index].pt)
    src_point = np.array(src_point, dtype=np.int)
    dst_point = np.array(dst_point, dtype=np.int)
    map, _ = cv.findHomography(dst_point, src_point, cv.RANSAC)
    # 计算四个角的变化后，因为warpPerspective函数是需要基于对应点的
    top_left = [0, 0, 1]
    bot_left = [0, shaped[0], 1]
    top_right = [shaped[1], 0, 1]
    bot_right = [shaped[1], shaped[0], 1]  # 注意这里所有点的坐标顺序是[长、宽、1]
    a_top_left = np.dot(map, top_left)
    a_bot_left = np.dot(map, bot_left)
    a_top_right = np.dot(map, top_right)
    a_bot_right = np.dot(map, bot_right)
    a_top_left = a_top_left/a_top_left[2]
    a_bot_left = a_bot_left / a_bot_left[2]
    a_top_right = a_top_right / a_top_right[2]
    a_bot_right = a_bot_right / a_bot_right[2]
    # print(bot_right, bot_left)
    # print(a_bot_right, a_bot_left)
    after = cv.warpPerspective(sub, map, (int(a_bot_right[0]), base.shape[0]))  # args:原图，映射参数，图片大小
    after = cv.warpPerspective(sub, map, (base.shape[1], base.shape[0]))
    cv.imshow('best', after)
    cv.imshow('base', base)
    # 计算图像并拷贝
    key_p = [src_point[0][0], src_point[0][1], 1]
    key_p_2 = [dst_point[0][0], dst_point[0][1], 1]  # 这个是同样的点在sub原图中的位置
    a_key_p = np.dot(map, key_p_2)
    a_key_p = a_key_p/a_key_p[2]
    width = int(key_p[0]+(after.shape[1]-a_key_p[0]))
    height = int(shaped[0])
    composite = np.zeros([height, width, 3], dtype=np.uint8)
    composite[:, 0:shaped[1]] = base
    composite[:, key_p[0]:width] = after[:, key_p[0]:after.shape[1]]
    print(key_p[0], width)
    cv.imshow('after', composite)
    cv.waitKey(0)


# 为外部调用提供的接口函数，要求输入的两张图片大小要保持一致
def pic_joint(base, sub, dir='LEFT'):
    shaped = sub.shape
    surf = cv.xfeatures2d.SURF_create()
    # 进行各自的特征点检测
    kp1, des1 = surf.detectAndCompute(base, None)
    kp2, des2 = surf.detectAndCompute(sub, None)
    # 匹配
    bf = cv.BFMatcher()
    good = []
    matches = bf.knnMatch(des1, des2, 2)
    for m, n in matches:
        if m.distance < 0.55 * n.distance:
            good.append([m])
    # 取前10个最佳匹配点
    check_num = 10  # 需要进行匹配的点个数
    temp = []
    best = []
    for index, d in enumerate(good):
        temp.append(d[0].distance)
    temp_sort = copy.copy(temp)
    temp_sort.sort()
    for i in range(check_num):
        index = temp.index(temp_sort[i])
        best.append(good[index])
    if dir == 'LEFT':
        show_best = cv.drawMatchesKnn(sub, kp2, base, kp1, good, None)
    else:
        show_best = cv.drawMatchesKnn(base, kp1, sub, kp2, best, None)
    cv.imshow('best', show_best)
    # 开始基于匹配点做放射变换
    src_point = []
    dst_point = []
    for i in range(10):
        src_index = best[i][0].queryIdx  # 用这个来获取原图中和目标图中的关键点索引
        dst_index = best[i][0].trainIdx
        src_point.append(kp1[src_index].pt)
        dst_point.append(kp2[dst_index].pt)
    src_point = np.array(src_point, dtype=np.int)
    dst_point = np.array(dst_point, dtype=np.int)
    map, _ = cv.findHomography(dst_point, src_point, cv.RANSAC)
    # 计算四个角的变化后
    top_left = [0, 0, 1]
    bot_left = [0, shaped[0], 1]
    top_right = [shaped[1], 0, 1]
    bot_right = [shaped[1], shaped[0], 1]
    a_top_left = np.dot(map, top_left)
    a_bot_left = np.dot(map, bot_left)
    a_top_right = np.dot(map, top_right)
    a_bot_right = np.dot(map, bot_right)
    a_top_left = a_top_left/a_top_left[2]
    a_bot_left = a_bot_left / a_bot_left[2]
    a_top_right = a_top_right / a_top_right[2]
    a_bot_right = a_bot_right / a_bot_right[2]
    after = cv.warpPerspective(sub, map, (int(a_bot_right[0]), base.shape[0]))
    # 计算图像并拷贝
    key_p = [src_point[0][0], src_point[0][1], 1]  # 这个是第一个关键点在base图中的位置
    key_p_2 = [dst_point[0][0], dst_point[0][1], 1]  # 这个是同样的点在sub原图中的位置
    a_key_p = np.dot(map, key_p_2)  # 这个是关键点在变换后的sub中的位置
    a_key_p = a_key_p/a_key_p[2]
    u_key_point = int(a_key_p[0])
    print('关键点横坐标', u_key_point)
    width = int(a_key_p[0]+(base.shape[1]-key_p[0]))  # 这个裁剪是：sub(0:关键点1所在的横向位置)+base(关键点1位置:end)
    height = int(shaped[0])
    composite = np.zeros([height, width, 3], dtype=np.uint8)
    composite[:, 0:u_key_point] = after[:, 0:u_key_point]
    composite[:, u_key_point:width] = base[:, key_p[0]:base.shape[1]]
    return composite


# 基于特征点匹配方法的自动标定技术
def pic_auto_cali():
    base = cv.imread('../img_lib/base.jpg')
    sub = cv.imread('../img_lib/left_1.jpg')
    base = cv.imread('../img_lib/low.jpg')
    sub = cv.imread('../img_lib/high.jpg')
    base = read.crop(base)
    sub = read.crop(sub)
    surf = cv.xfeatures2d.SURF_create()
    # 进行各自的特征点检测
    kp1, des1 = surf.detectAndCompute(base, None)
    kp2, des2 = surf.detectAndCompute(sub, None)
    # 匹配，具体匹配的参数描述详见 https://blog.csdn.net/weixin_44072651/article/details/89262277
    bf = cv.BFMatcher()
    good = []
    matches = bf.knnMatch(des1, des2, 2)
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good.append([m])
    show_good = cv.drawMatchesKnn(base, kp1, sub, kp2, good, None)
    # cv.imshow('hhh', show_good)
    # cv.waitKey(0)
    # 取前10个最佳匹配点
    check_num = 10  # 需要进行匹配的点个数
    temp = []
    best = []
    for index, d in enumerate(good):  # 这里使用了遍历操作，index表示索引，d才是good中的match点
        temp.append(d[0].distance)  # 排列刚刚good中的欧氏距离，做进一步筛选
    temp_sort = copy.copy(temp)
    temp_sort.sort()
    for i in range(check_num):
        index = temp.index(temp_sort[i])
        best.append(good[index])  # 这里把good进一步到best
    show_best = cv.drawMatchesKnn(base, kp1, sub, kp2, best, None)  # 再进行绘制
    # cv.imshow('best', show_best)
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
        print('No.', i, ': x_gap ', src_point[i][0]-dst_point[i][0], ', y_gap ', src_point[i][1]-dst_point[i][1])
    x_mean_differ = int(np.mean(x_differ))
    y_mean_differ = int(np.mean(y_differ))
    #  下面这些点的形式是x,y（注意！）
    src_point = np.array(src_point, dtype=np.int)
    dst_point = np.array(dst_point, dtype=np.int)
    print('x', x_mean_differ, y_mean_differ)
    new_sub = np.zeros([sub.shape[0], sub.shape[1], 3], np.uint8)
    new_sub[0:y_mean_differ+sub.shape[0], :] = sub[-y_mean_differ:sub.shape[0], :]  # 这是向上移动
    # new_sub[-y_mean_differ:sub.shape[0], :] = sub[0:y_mean_differ+sub.shape[0], :]  # 向下移动
    cv.imshow('new_sub', new_sub)
    cv.imshow('origin', base)
    cv.waitKey(0)
    # #########################################################################
    # # 计算旋转角度，通过选取src和dst的对应两点之间的连线的角度值进行矫正(取10次连线)
    # # 1、首先选取两点距离较长的两个点作为基准，有一个while判定函数，以下50设置为线长阈值
    # # 2、用向量夹角公式计算出相应夹角
    # theta = []  # cos theta 坐标格式
    # for i in range(10):
    #     import random
    #     line_length = 0
    #     while line_length < 50:  # 索引过小会导致两点连线过近，于是需要限制
    #         a = random.randint(0, check_num-1)
    #         b = random.randint(0, check_num-1)
    #         # 这是测试线，要保证测试线的足够长才能保证计算的角度准确，否则继续while循环找距离较远的相邻两组
    #         test_line = [src_point[a][0] - src_point[b][0], src_point[a][1] - src_point[b][1]]
    #         line_length = math.sqrt(test_line[0] ** 2 + test_line[1] ** 2)  # 线长
    #         print('线长为：', line_length)
    #     # 以上工作是取两个不同索引下标，以下工作是取得原图和目标图中的两个连线坐标
    #     src_line = [src_point[a][0] - src_point[b][0], src_point[a][1] - src_point[b][1]]  # 计算坐标形式下的直线
    #     dst_line = [dst_point[a][0] - dst_point[b][0], dst_point[a][1] - dst_point[b][1]]
    #     # 以下是计算内积（dot）和绝对值乘积用来运算cos theta值
    #     dot = src_line[0]*dst_line[0] + src_line[1]*dst_line[1]  # 内积公式 x1*x2+y1*y2
    #     #  绝对值乘积公式 |a|*|b|
    #     abs = math.sqrt(src_line[0] ** 2 + src_line[1] ** 2) * math.sqrt(dst_line[0] ** 2 + dst_line[1] ** 2)
    #     theta.append(math.acos(dot/abs)*180/math.pi)  # math自带的是弧度制
    #     # 计算角度
    #     print(a, b, theta[i])
    #     # 画原图上的线
    #     cv.circle(base, (src_point[a][0], src_point[a][1]), 5, (255, 100, 10), -1)
    #     cv.circle(base, (src_point[b][0], src_point[b][1]), 5, (255, 100, 10), -1)
    #     cv.line(base, (src_point[a][0], src_point[a][1]), (src_point[b][0], src_point[b][1]), (255, 255, 0), 2)
    #     # 画dst上的线
    #     cv.circle(sub, (dst_point[a][0], dst_point[a][1]), 5, (255, 100, 10), -1)
    #     cv.circle(sub, (dst_point[b][0], dst_point[b][1]), 5, (255, 100, 10), -1)
    #     cv.line(sub, (dst_point[a][0], dst_point[a][1]), (dst_point[b][0], dst_point[b][1]), (255, 255, 0), 2)
    #     cv.imshow('base', base)
    #     cv.imshow('sub', sub)
    #     cv.waitKey(0)
    # angle = np.mean(theta)
    # print('平均旋转角度：', angle)

    map, _ = cv.findHomography(dst_point, src_point, cv.RANSAC)


if __name__ == '__main__':
    # pic_auto_cali()
    pic_joint_test()
