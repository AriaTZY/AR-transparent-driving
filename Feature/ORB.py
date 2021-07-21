# Oriented FAST and Rotated BRIEF方法
import cv2 as cv
from Feature import read


# 使用基于FAST和BRIEF表述方法的角点检测，非常快
# 但是FAST没有方向性，且精准度不高。所以使用Harris又进行角点筛选
# 又用了一些方法进行方向确定，最终用BRIEF表述
def ORB():
    pic = read.get_pics(0, 'Lena')
    orb = cv.ORB_create()
    kp, _ = orb.detectAndCompute(pic, None)
    show = cv.drawKeypoints(pic, kp, None)
    cv.imshow('ORB', show)
    print(len(kp))
    cv.waitKey(0)


# 经过检测，可得FAST得到的角点会最多最杂；其次是SURF和SIFT的检测，因为他们都属于特征点检测，所以不局限于角点，所以点多
# SHI和ORB的检测都差不多，但Harris角点检测会检测出较多边缘和堆积在一起的地方。ORB虽属于特征点检测，但其实跟多的是角点
if __name__ == '__main__':
    ORB()
