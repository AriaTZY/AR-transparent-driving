import cv2 as cv
from UI_support import part_joint, y_offset_pic
import read_support as read
import math

# cap_phone = cv.VideoCapture('video/iphone_2.mp4')
# cap_huawei = cv.VideoCapture('video/huawei_2.mp4')
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter('testwrite.avi', fourcc, 20.0, (642, 385), True)
# yaw = 0
# pitch = 0
#
# for i in range(70):
#     ret, frame1 = cap_phone.read()
#     ret, frame2 = cap_huawei.read()
# for i in range(1000):
#     ret, frame1 = cap_phone.read()
#     ret, frame2 = cap_huawei.read()
#     frame1 = cv.resize(frame1, (0, 0), None, 0.5, 0.5)
#     frame2 = cv.resize(frame2, (0, 0), None, 0.5, 0.5)
#     frame2 = cv.flip(frame2, 0)
#     frame2 = cv.flip(frame2, 1)
#     left = read.crop(frame2)
#     right = read.crop(frame1)
#     # cv.imwrite('video/left.jpg', left)
#     # cv.imwrite('video/right.jpg', right)
#     left = y_offset_pic(left, 7)
#     if 600 > i > 300:
#         yaw = int(25*math.sin((i-300)/150*math.pi))
#     if 150 > i > 100:
#         pitch = int(10*math.sin((i-100)/25*math.pi))
#     joint = part_joint(left, right, -33, pitch, yaw)
#     cv.imshow('right', joint)
#     cv.waitKey(1)
#     print(joint.shape)
#     out.write(joint)
#     # cv.waitKey(0)
# out.release()


def mouse_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONUP:
        print(y)


def draw_tax_on():
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output_m3.avi', fourcc, 50.0, (642, 385), True)
    cap_phone = cv.VideoCapture('testwrite.avi')
    ret, pic = cap_phone.read()
    import xlrd
    wb = xlrd.open_workbook('car_position.xls')  # 打开文件
    sheet1 = wb.sheet_by_index(0)  # 通过索引获取表格
    count = 0
    while ret:
        rows = sheet1.row_values(count)  # 获取行内容
        num = rows[0]
        for i in range(int(num)):
            x = int(rows[i * 4 + 1])
            y = int(rows[i * 4 + 2])
            w = int(rows[i * 4 + 3])
            h = int(rows[i * 4 + 4])
            # if 149 > count > 123:  # 这一部分是头向上仰，所以就会造成误检测
            if (y + h) > 259 and not (149 > count > 123):
                cv.rectangle(pic, (x - w, y - h), (x + w, y + h), (0, 0, 255), 4)
                cv.rectangle(pic, (x - w-2, y - h - 20), (x + w+2, y - h), (200, 200, 200), -1)
                cv.putText(pic, 'Caution!', (x-w, y-h), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv.rectangle(pic, (x - w, y - h), (x + w, y + h), (0, 255, 0), 1)
        cv.imshow('Window', pic)
        cv.setMouseCallback('Window', mouse_event)
        cv.waitKey(1)
        print('frame', count)
        out.write(pic)
        ret, pic = cap_phone.read()
        count += 1
    out.release()


if __name__ == '__main__':
    draw_tax_on()
