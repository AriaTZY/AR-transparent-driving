import cv2 as cv
import random
import os.path
import numpy as np
import pylab


flag_draw = 0
start_point = (0, 0)
end_point = (0, 0)


# 在图像中画出正方形，就可以自动保存
def draw(event, x, y, flags, param):
    global start_point
    global flag_draw
    global show
    global pic
    if event == cv.EVENT_LBUTTONDOWN:
        flag_draw = 1
        start_point = (x, y)
    elif event == cv.EVENT_LBUTTONUP:
        flag_draw = 0
        end_point = (x, y)
        # 裁剪其他位置
        width = end_point[0] - start_point[0]
        height = end_point[1] - start_point[1]
        cv.rectangle(show, start_point, end_point, (255, 0, 255), 3)
        index_name = random.randint(0, 50000)
        path = 'D:\\show_pic\\car_detection\\neg\\' + str(index_name) + '.jpg'
        ROI = pic[start_point[1]:end_point[1], start_point[0]:end_point[0]]  # 这个是自定义长宽，用于裁剪正样本
        ROI = pic[start_point[1]:start_point[1]+int(width/1.5), start_point[0]:end_point[0]]  # 根据宽固定长，用于裁剪负样本
        cv.imwrite(path, ROI)
        div = 5
        div = 2
        for x_off in range(-int(width/div), int(width/div), 5):
            for y_off in range(-int(height / div), int(height / div), 5):
                t_start_point = (start_point[0]-x_off, start_point[1]-y_off)
                t_end_point = (end_point[0] - x_off, end_point[1] - y_off)
                cv.rectangle(show, t_start_point, t_end_point, (0, 0, 255), 1)
                index_name = random.randint(0, 50000)
                path = 'D:\\show_pic\\car_detection\\neg\\' + str(index_name) + '.jpg'
                ROI = pic[t_start_point[1]:t_end_point[1], t_start_point[0]:t_end_point[0]]
                ROI = pic[t_start_point[1]:t_start_point[1] + int(width / 1.5), t_start_point[0]:t_end_point[0]]
                cv.imwrite(path, ROI)
        print('左键按下')


# 对视频进行保存为图片的函数
def save_picture(start=10):
    cap = cv.VideoCapture('G:\\video\\DCIMA\\20190221_031459A.mp4')
    count = start
    if not cap.isOpened():
        print('视频打开错误')
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv.resize(frame, (640, 360), None)
            cv.imshow('Frame', frame)
            # cv.setMouseCallback('Frame', draw)
            a = cv.waitKey(0)
            if a == 115:  # 115是save的意思
                name = 'D:\\show_pic\\car_detection\\' + str(count) + '.jpg'
                cv.imwrite(name, frame)
                count = count + 1
                print('保存成功')
        else:
            break
    print('视频播放完毕')


# 显示裁剪好的图片，并与回调函数进行配合操作裁剪图片
def crop_car(start=13):
    global show
    global pic
    for i in range(start, 139):
        path = 'D:\\show_pic\\car_detection\\' + str(i) + '.jpg'
        pic = cv.imread(path, 0)  # 这是真实裁剪的图
        show = cv.imread(path, 1)  # 这只是用来显示的图
        while True:
            cv.imshow('hhh', show)
            cv.setMouseCallback('hhh', draw)
            a = cv.waitKey(1)
            if a == 112:
                break


# 对初步裁剪好的图片进行图像大小归一化
def normalized_car():
    rootdir = 'D:\\show_pic\\car_detection\\neg'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    # times = 0.
    # count = 0
    # for i in range(len(list)):
    #     temp = cv.imread(os.path.join(rootdir, list[i]), 0)
    #     if temp is None:
    #         print('open error')
    #     else:
    #         count = count + 1
    #         times = times + temp.shape[1]/temp.shape[0]
    # print('avr_time:', times/count)
    # 经过以上的统计，我们大概把宽度统一为60，长宽比为1.50
    count = 0
    for i in range(len(list)):
        temp = cv.imread(os.path.join(rootdir, list[i]), 0)
        if temp is None:
            print('open error')
        else:
            temp = cv.resize(temp, (60, 40))
            path = 'D:\\show_pic\\car_detection\\neg\\normalized\\' + str(count) + '.jpg'
            cv.imwrite(path, temp)
            count = count + 1


# 对所有图片再进行不同尺度的处理
def scale_process(end_index=5098):
    path = 'D:\\show_pic\\car_detection\\neg\\normalized\\'
    count = 0
    for index in range(end_index):
        name = path + str(index) + '.jpg'
        pic = cv.imread(name, 0)
        if pic is not None:
            # 0.8的尺度变换
            temp = cv.resize(pic, (0, 0), None, 0.8, 0.8)
            temp = cv.resize(temp, (60, 40), None)
            save_name = path + str(count + end_index) + '.jpg'
            count = count + 1
            cv.imwrite(save_name, temp)
            # 0.6的尺度变换
            temp = cv.resize(pic, (0, 0), None, 0.6, 0.6)
            temp = cv.resize(temp, (60, 40), None)
            save_name = path + str(count + end_index) + '.jpg'
            count = count + 1
            cv.imwrite(save_name, temp)


# 把图片转换成numpy数组，并打乱保存。保存后的数组会占据很大空间，要计时删除
def cvt2numpy(train_pos=200, train_neg=100, test=10):
    train = train_neg + train_pos
    path_pos = 'D:\\show_pic\\car_detection\\front\\normalized\\'
    path_neg = 'D:\\show_pic\\car_detection\\neg\\normalized\\'
    train_np = np.zeros([train, 60*40], dtype=np.float32)
    test_np = np.zeros([test, 60 * 40], dtype=np.float32)
    train_label = np.zeros([train, 2], dtype=np.float32)
    test_label= np.zeros([test, 2], dtype=np.float32)
    train_count = 0
    test_count = 0
    # 读取train（分两部分pos，neg读入）
    for i in range(train_pos):
        name = path_pos + str(i) + '.jpg'
        pic = cv.imread(name, 0)
        if pic is not None:
            train_np[train_count] = np.reshape(pic, [1, 2400])
            train_label[train_count] = [1, 0]
            train_count = train_count + 1
        else:
            print('打开失败！', name)
    for i in range(train_neg):
        name = path_neg + str(i) + '.jpg'
        pic = cv.imread(name, 0)
        if pic is not None:
            train_np[train_count] = np.reshape(pic, [1, 2400])
            train_label[train_count] = [0, 1]
            train_count = train_count + 1
        else:
            print('打开失败！', name)
    # 读取test（分两部分pos，neg读入）
    for i in range(train // 2, train // 2 + test//2):
        name = path_pos + str(i) + '.jpg'
        pic = cv.imread(name, 0)
        if pic is not None:
            test_np[test_count] = np.reshape(pic, [1, 2400])
            test_label[test_count] = [1, 0]
            test_count = test_count + 1
        else:
            print('打开失败！', name)
    for i in range(train // 2, train // 2 + test//2):
        name = path_neg + str(i) + '.jpg'
        pic = cv.imread(name, 0)
        if pic is not None:
            test_np[test_count] = np.reshape(pic, [1, 2400])
            test_label[test_count] = [0, 1]
            test_count = test_count + 1
        else:
            print('打开失败！', name)

    # 现在开始打乱顺序
    for i in range(train):
        wait = random.randint(0, train-1)
        temp1 = train_np[wait]
        temp2 = train_label[wait]
        train_np[wait] = train_np[i]
        train_label[wait] = train_label[i]
        train_np[i] = temp1
        train_label[i] = temp2
    for i in range(test):
        wait = random.randint(0, test-1)
        temp1 = test_np[wait]
        temp2 = test_label[wait]
        test_np[wait] = test_np[i]
        test_label[wait] = test_label[i]
        test_np[i] = temp1
        test_label[i] = temp2
    print(test_label)
    np.save('data/train_data', train_np)
    np.save('data/train_label', train_label)
    np.save('data/test_data', test_np)
    np.save('data/test_label', test_label)


# 抽插label和图片是否对应的（肉眼人工检查）
def check_right(start=90):
    data = np.load('data/train_data.npy')
    label = np.load('data/train_label.npy')
    count = 1
    for i in np.arange(start, start+20, 1):
        pylab.subplot(4, 5, count)
        show_pic = np.reshape(data[i], [40, 60])
        pylab.imshow(show_pic, 'gray')
        word = 'num=' + str(label[i])
        pylab.title(word)
        pylab.axis('off')
        count = count + 1
    pylab.show()


if __name__ == '__main__':
    # crop_car(start=0)
    # normalized_car()
    # scale_process(end_index=9356)
    # check_right(start=90)
    # cvt2numpy(train_pos=16000, train_neg=27500, test=200)
    cvt2numpy(train_pos=16, train_neg=27, test=2)
