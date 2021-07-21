import cv2 as cv
from Feature import read


# 使用基于像素圆（16pix）的角点检测，非常快
def FAST():
    pic = read.get_pics(0, 'Lena')
    fast = cv.FastFeatureDetector_create()
    kp = fast.detect(pic, None)
    show = cv.drawKeypoints(pic, kp, None)
    cv.imshow('FAST_with_Non_MaxSuppression', show)
    print(len(kp))
    # 原本的FAST中就有非极大值抑制检测，这里手动关闭，查看对比效果，默认值大概也就10左右，再网上不会有变化了
    fast.setNonmaxSuppression(0)
    kp = fast.detect(pic, None)
    show = cv.drawKeypoints(pic, kp, None)
    print(len(kp))
    cv.imshow('FAST_without_Non_MaxSuppression', show)
    # with brief描述子
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    kp, des = brief.compute(pic, kp)
    show = cv.drawKeypoints(pic, kp, None)
    cv.imshow('FAST_with_Brief', show)
    cv.waitKey(0)


if __name__ == '__main__':
    FAST()
