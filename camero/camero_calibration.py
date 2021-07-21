import cv2 as cv
import numpy as np


def camero_cali(h=6, w=7):
    for i in range(13):
        if i < 9:
            path = 'D:/opencv/sources/samples/data/left0' + str(i+1) + '.jpg'
        else:
            path = 'D:/opencv/sources/samples/data/left' + str(i+1) + '.jpg'
        pic = cv.imread(path, 0)
        ret, corners = cv.findChessboardCorners(pic, (h, w), None)
        # 定义在真实坐标系下的棋盘格坐标，统一设置z=0，小正方形的边长默认设置为 1
        objp = np.zeros((w * h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)
        # 列表的point
        real_world_p = []
        ima_p = []
        if ret:
            print('检测成功！可以进行标定！')
            ima_p.append(corners)
            real_world_p.append(objp)
            for i in range(w*h):
                i, j = corners[i][0]
                cv.circle(pic, (i, j), 3, 255, 3)
    # 标志，matrix相机矩阵，畸变参数，rotation向量，translation向量
    # dist - k1, k2, p1, p2, k3
    # 注意这里opencv的size输入是(长-高)而numpy中是(高-长)，所以要进行转换一下
    pic = cv.imread('D:/opencv/sources/samples/data/left04.jpg', 0)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(real_world_p, ima_p, pic.shape[::-1], None, None)
    # 0表示直接裁剪好
    new_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, pic.shape[::-1], 0, pic.shape[::-1])
    dst = cv.undistort(pic, new_mtx, dist)
    print('mtx:', mtx)
    print('new_mtx', new_mtx)
    count = 0
    while True:
        if count % 2 == 0:
            cv.imshow('hhh', dst)
        else:
            cv.imshow('hhh', pic)
        count += 1
        cv.waitKey(30000)


# 此函数是为广角160度摄像头的畸变校正函数
def cali_live(pic):
    # 相机内参
    mtx = np.array([[405.577347304559, -0.661209154905884, 260.569357581653],
                    [0, 404.373958635926, 253.066918158317],
                    [0, 0, 1]])
    dist = np.array([-0.323485229902059, 0.103289325466282, 0.00158254706232522, 0.00249055910248026, 0])

    shaped = pic.shape
    # alpha是0保留最小区域，为1还会有黑边，但会保留所有像素，其实这一步不用也行
    new_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (shaped[1], shaped[0]), alpha=0)
    dst = cv.undistort(pic, mtx, dist, None, new_mtx)
    return dst


if __name__ == '__main__':
    camero_cali()
