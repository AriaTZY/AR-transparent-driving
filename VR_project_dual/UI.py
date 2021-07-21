import cv2 as cv
import read_support as read
from UI_support import *
from perspective_spport import serial_control
import xlrd
import xlwt


current_page = 0
button_index = 0  # 表示目前按压下的按键
left_yaw = 20
left_pitch = 0  # 这是左右两个图片得上下平移量，负数表示左右图片应该相应向上平移
view_yaw = 50


# 函数是用来对点击下的坐标进行具体是哪一个按键的确定，给入x,y坐标，返回按键index
def check_btn(x, y):
    btn_1 = [[32, 112], [398, 194]]  # 这个是按钮的坐标，分别为[x, y]格式，左上、右下
    btn_2 = [[25, 203], [358, 281]]
    btn_3 = [[27, 300], [314, 379]]
    btn_4 = [[30, 15], [125, 84]]
    if (btn_1[1][1] > y > btn_1[0][1]) and (btn_1[1][0] > x > btn_1[0][0]):
        # print('pass button 1')
        return 1
    elif (btn_2[1][1] > y > btn_2[0][1]) and (btn_2[1][0] > x > btn_2[0][0]):
        # print('pass button 2')
        return 2
    elif (btn_3[1][1] > y > btn_3[0][1]) and (btn_3[1][0] > x > btn_3[0][0]):
        # print('pass button 3')
        return 3
    elif (btn_4[1][1] > y > btn_4[0][1]) and (btn_4[1][0] > x > btn_4[0][0]):
        # print('pass button 4')
        return 4
    else:
        return 0


# 进行坐标校准，主要是在按键初步确定坐标的时候用到，输出左上角和右下角的坐标
def button_position_calibration(width):
    test = cv.imread('../img_lib/UI/mainpage_default.jpg')
    test = read.crop2(test, standard_width)
    cv.imshow('SHOW', test)
    cv.setMouseCallback('SHOW', mouse_event_for_cali)


# 专门给坐标确定函数专用的回掉函数
def mouse_event_for_cali(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONUP:
        print(x, y)


def mouse_event(event, x, y, flags, param):
    global current_page, button_index
    if event == cv.EVENT_LBUTTONUP:
        if current_page == 0:  # 在主菜单的操作过程
            if button_index == 1:
                print('进入子菜单')
                # cv.destroyWindow('VR Project')
                cv.imshow('Manual Calibration', pic_mid)  # 在变化的初始的过程中，需要做show和贴上进度条的工作
                cv.createTrackbar('View Yaw', 'Manual Calibration', 0, 100, nothing)
                cv.createTrackbar('Yaw', 'Manual Calibration', 0, 50, nothing)
                # 再设置一些默认值
                cv.setTrackbarPos('View Yaw', 'Manual Calibration', 50)
                cv.setTrackbarPos('Yaw', 'Manual Calibration', int(left_yaw))
                current_page = 1
            elif button_index == 2:
                print('进入子菜单2')
                cv.destroyWindow('Auto Calibration')
                current_page = 2
            elif button_index == 4:
                serial_control(-left_yaw, left_pitch)
                current_page = 0
            elif button_index == 3:
                serial_control_rocker(-left_yaw, -left_pitch)
                current_page = 0
            button_index = 0
    if event == cv.EVENT_MOUSEMOVE:
        if current_page == 0:  # 在主界面才会进行鼠标按压下的动作
            button_index = check_btn(x, y)


def nothing(x):
    pass


def auto_cali_Trackbar(x):
    global view_yaw
    view_yaw = x


# 在主界面的UI上显示当前参数
def draw_params(pic):
    img = copy.copy(pic)
    size = img.shape
    cv.rectangle(img, (size[1]-200, 45), (size[1]-30, 110), (255, 255, 255), -1)
    cv.putText(img, 'Dual Camera Mode', (size[1]//2, 30), cv.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 200), 2)  # 字高大约10pix
    cv.putText(img, 'Params:', (size[1]-200+5, 60), cv.FONT_HERSHEY_COMPLEX, 0.5, 0, 1)  # 字高大约10pix
    cv.putText(img, ' Yaw:' + str(int(left_yaw)), (size[1] - 200 + 10, 75), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
    cv.putText(img, ' Hei:' + str(int(left_pitch)), (size[1] - 200 + 10, 90), cv.FONT_HERSHEY_COMPLEX, 0.5, (25, 80, 100), 1)
    return img


# 保存参数，参数分别为左、右两个校准值
def write_excel(left, left_p):
    f = xlwt.Workbook()
    print('保存中：', left, left_p)
    sheet1 = f.add_sheet('manual_set', cell_overwrite_ok=True)
    row0 = [left, left_p]
    # 写第一行
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i])
    f.save('info.xls')


def read_excel():
    global left_yaw, right_yaw, left_pitch, right_pitch
    wb = xlrd.open_workbook('info.xls')  # 打开文件
    sheet1 = wb.sheet_by_index(0)  # 通过索引获取表格
    rows = sheet1.row_values(0)  # 获取行内容
    left_yaw = rows[0]
    left_pitch = rows[1]
    print('参数读取成功！')


# 这是一个过场动画，显示saving...
def saving_process(pic, win_name):
    text = 'Saving'
    origin_joint = copy.copy(pic)
    for i in range(5):
        cv.putText(pic, text, (pic.shape[1] // 3, pic.shape[0] // 2), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                   (255, 0, 255), 2)
        cv.imshow(win_name, pic)
        cv.waitKey(100)
        text = text + '.'
    cv.putText(origin_joint, 'Success!', (pic.shape[1] // 3, pic.shape[0] // 2), cv.FONT_HERSHEY_SIMPLEX, 1.2,
               (255, 0, 255), 2)
    cv.imshow(win_name, origin_joint)
    cv.waitKey(1000)


if __name__ == '__main__':
    read_excel()
    standard_width = 900  # 界面标准宽度设置
    # button_position_calibration(standard_width)
    main_page = cv.imread('img_lib/UI/mainpage_default.jpg')
    main_page_btn1 = cv.imread('img_lib/UI/mainpage_btn1.jpg')
    main_page_btn2 = cv.imread('img_lib/UI/mainpage_btn2.jpg')
    main_page_btn3 = cv.imread('img_lib/UI/mainpage_btn3.jpg')
    main_page_btn4 = cv.imread('img_lib/UI/mainpage_start.jpg')
    pic_mid = cv.imread('img_lib/base.jpg')
    pic_left = cv.imread('img_lib/left_3.jpg')
    pic_right = cv.imread('img_lib/right_2.jpg')
    cap1 = cv.VideoCapture(cfg.camera_order[0])
    cap2 = cv.VideoCapture(0)
    cap3 = cv.VideoCapture(cfg.camera_order[1])
    # 做图片限幅重置大小处理
    main_page = read.crop2(main_page, standard_width)
    main_page_btn1 = read.crop2(main_page_btn1, standard_width)
    main_page_btn2 = read.crop2(main_page_btn2, standard_width)
    main_page_btn3 = read.crop2(main_page_btn3, standard_width)
    main_page_btn4 = read.crop2(main_page_btn4, standard_width)
    pic_mid = read.crop(pic_mid)
    pic_left = read.crop(pic_left)
    pic_right = read.crop(pic_right)
    while True:
        while current_page == 0:
            if button_index == 0:
                show = draw_params(main_page)
                cv.imshow('VR Project', show)
            elif button_index == 1:
                show = draw_params(main_page_btn1)
                cv.imshow('VR Project', show)
            elif button_index == 2:
                show = draw_params(main_page_btn2)
                cv.imshow('VR Project', show)
            elif button_index == 3:
                show = draw_params(main_page_btn3)
                cv.imshow('VR Project', show)
            elif button_index == 4:
                show = draw_params(main_page_btn4)
                cv.imshow('VR Project', show)
            cv.setMouseCallback('VR Project', mouse_event, current_page)
            cv.waitKey(1)
        while current_page == 1:
            # view_yaw形参传递进去 正值-向左  负值-向右
            # 在left_yaw中，负得越多表示向左偏转
            # 在right_yaw中，负得越多表示向右偏转
            # 读入图片
            pic_left, _, pic_mid = read.video_pic_cap(cap1, cap2, cap3)  # 读取图像
            pic_left1 = y_offset_pic(pic_left, left_pitch)
            joint = part_joint(pic_left1, pic_mid, -left_yaw, 10, -view_yaw+50)
            cv.imshow('Manual Calibration', joint)
            left_yaw = cv.getTrackbarPos('Yaw', 'Manual Calibration')
            view_yaw = cv.getTrackbarPos('View Yaw', 'Manual Calibration')
            a = cv.waitKey(1)
            if a == 98:  # 98是b，也就是back的缩写
                cv.destroyWindow('Manual Calibration')
                current_page = 0
            elif a == 115:  # 115是s，也就是save的缩写
                cv.destroyWindow('Manual Calibration')
                cv.imshow('Manual Calibration 2', joint)  # 在变化的初始的过程中，需要做show和贴上进度条的工作
                cv.createTrackbar('View Yaw', 'Manual Calibration 2', 0, 100, nothing)
                cv.createTrackbar('Height', 'Manual Calibration 2', 0, 60, nothing)  # 正负+-20
                # # 再设置一些默认值
                cv.setTrackbarPos('View Yaw', 'Manual Calibration 2', 50)
                cv.setTrackbarPos('Height', 'Manual Calibration 2', int(left_pitch+30))
                key_val = 0
                while True:
                    pic_left, _, pic_mid = read.video_pic_cap(cap1, cap2, cap3)
                    left_pitch_temp = cv.getTrackbarPos('Height', 'Manual Calibration 2')
                    view_yaw = cv.getTrackbarPos('View Yaw', 'Manual Calibration 2')
                    # 进行一下换算
                    left_pitch = left_pitch_temp - 30
                    pic_left1 = y_offset_pic(pic_left, left_pitch)
                    joint = part_joint(pic_left1, pic_mid, -left_yaw, 10, -view_yaw + 50)
                    cv.imshow('Manual Calibration 2', joint)
                    key_val = cv.waitKey(1)
                    if key_val == 115:
                        # 保存参数
                        write_excel(left_yaw, left_pitch)
                        saving_process(joint, 'Manual Calibration 2')
                        cv.destroyWindow('Manual Calibration 2')
                        break
                if key_val == 115:
                    current_page = 0
                    break
        while current_page == 2:
            for i in range(10):
                _, _, _ = read.video_pic_cap(cap1, cap2, cap3)  # 读取图像，第一帧图像往往偏黑，所以空读取一次
            pic_left, _, pic_mid = read.video_pic_cap(cap1, cap2, cap3)  # 读取图像
            x_gap_L1, y_gap_L1, _ = pic_auto_cali(pic_left, pic_mid, 'Auto Calibration')
            pic_left, _, pic_mid = read.video_pic_cap(cap1, cap2, cap3)  # 读取图像
            x_gap_L2, y_gap_L2, _ = pic_auto_cali(pic_left, pic_mid, 'Auto Calibration')
            pic_left, _, pic_mid = read.video_pic_cap(cap1, cap2, cap3)  # 读取图像
            x_gap_L3, y_gap_L3, show_pic = pic_auto_cali(pic_left, pic_mid, 'Auto Calibration')
            x_gap_L = (x_gap_L1 + x_gap_L2 + x_gap_L3)//3
            y_gap_L = (y_gap_L1 + y_gap_L2 + y_gap_L3) // 3
            print('像素角度是', x_gap_L)
            gain = 43/-320  # 这是一个像素点和角度得对应值，和摄像头得型号有关，需要进行标定
            left_yaw = gain * x_gap_L
            write_excel(left_yaw, y_gap_L)
            saving_process(show_pic, 'Auto Calibration')
            # 做完标定紧接着就开始进行显示
            joint = part_joint(pic_left, pic_mid, -left_yaw, 10, -view_yaw + 50)
            cv.imshow('Auto Calibration', joint)
            cv.createTrackbar('View Yaw', 'Auto Calibration', 0, 100, auto_cali_Trackbar)
            cv.setTrackbarPos('View Yaw', 'Auto Calibration', 50)
            while True:
                pic_left, _, pic_mid = read.video_pic_cap(cap1, cap2, cap3)  # 读取图像
                pic_left1 = y_offset_pic(pic_left, y_gap_L)  # 进行上下移动照片
                joint = part_joint(pic_left1, pic_mid, -left_yaw, 10, -view_yaw + 50)
                cv.imshow('Auto Calibration', joint)
                a = cv.waitKey(1)
                if a == 98:  # 98是b，也就是back的缩写
                    break
            if a == 98:  # 98是b，也就是back的缩写
                cv.destroyWindow('Auto Calibration')
                current_page = 0

