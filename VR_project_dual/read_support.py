import cv2 as cv
from matrix_support import cali_live

def get_pics(mode=0, name='Baboon'):
    path = 'D:/show_pic/'
    if name == 'Baboon':
        path = path+'Baboon.jpg'
        pic = cv.imread(path, mode)
    elif name == 'Lena':
        path = path + 'Lena.jpg'
        pic = cv.imread(path, mode)
    elif name == 'Airplane':
        path = path + 'Airplane.jpg'
        pic = cv.imread(path, mode)
    elif name == 'Fruit':
        path = path + 'Fruit.jpg'
        pic = cv.imread(path, mode)
    else:
        path = path + name + '.jpg'
        pic = cv.imread(path, mode)
    # 做限制大小的操作
    width = pic.shape[1]
    height = pic.shape[0]
    wide = max(width, height)
    if wide > 512:
        print('Too large, resize')
        scale = 512./wide
        pic = cv.resize(pic, (0, 0), pic, scale, scale)
    return pic


def crop(pic):
    width = pic.shape[1]
    height = pic.shape[0]
    wide = max(width, height)
    if wide > 512:
        # print('Too large, resize')
        scale = 512. / wide
        pic = cv.resize(pic, (0, 0), pic, scale, scale)
    return pic


# 这个是可以自定义宽度的crop函数
def crop2(pic, user_width=512):
    width = pic.shape[1]
    height = pic.shape[0]
    wide = max(width, height)
    # print('resize')
    scale = user_width / wide
    pic = cv.resize(pic, (0, 0), pic, scale, scale)
    return pic


# 做包括图片裁剪、图片畸变矫正的工作
def video_pic_cap(cap1, cap2, cap3):
    success1, pic1 = cap1.read()
    success2, pic2 = cap2.read()
    success3, pic3 = cap3.read()
    pic1 = pic1[60:420, :]  # opencv的读入格式长宽比有些问题，所以重新裁剪
    pic2 = pic2[60:420, :]
    pic3 = pic3[60:420, :]
    pic1 = cali_live(pic1)
    pic2 = cali_live(pic2)
    pic3 = cali_live(pic3)
    return pic1, pic2, pic3

