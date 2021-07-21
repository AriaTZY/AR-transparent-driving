import numpy as np
import cv2 as cv
import math
import pylab
# 本py文件是绘制二维坐标系下的点与点的坐标关系

# 创建w和c关系的矩阵，方向c=R*w，角度输入为角度制
def cal_RT(point=(30, 30, 1), angle=30., x_t=3.):
    angle = angle/180.*math.pi
    show = np.zeros([500, 500, 3], np.uint8)
    center = (250, 250)
    # 绘制基坐标系
    cv.line(show, (250, 250), (500, 250), (255, 255, 255), 2)
    cv.line(show, (250, 250), (250, 0), (255, 255, 255), 2)
    cv.circle(show, (250, 250), 5, (255, 255, 0), -1)
    # 绘制旋转后坐标系
    x_end = (500, int(250-250*math.sin(angle)))
    y_end = (int(250-250*math.sin(angle)), 0)
    cv.line(show, center, x_end, (255, 255, 0), 2)
    cv.line(show, center, y_end, (255, 255, 0), 2)
    cv.circle(show, (250, 250), 5, (255, 255, 0), -1)

    # 创建单体矩阵
    Rz = np.zeros([3, 3], dtype=np.float)
    # 计算绕x旋转矩阵
    Rz[0][0] = math.cos(angle)
    Rz[0][1] = math.sin(angle)
    Rz[1][0] = -math.sin(angle)
    Rz[1][1] = math.cos(angle)
    Rz[2][2] = 1.
    print(Rz)
    out = np.dot(Rz, point)
    print(out)

    cv.circle(show, (center[0]+point[0], center[0]-point[1]), 8, (255, 0, 0), -1)
    cv.circle(show, (center[0] + int(out[0]), center[0] - int(out[1])), 8, (0, 255, 0), -1)

    pylab.imshow(show)
    pylab.axis('off')
    pylab.show()


if __name__ == '__main__':
    cal_RT((70, 80, 0))

