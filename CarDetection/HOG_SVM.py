import cv2 as cv
import numpy as np
import pylab


def compute_HOG(ima_list):
    print('总数：', ima_list.shape[0])
    hog = cv.HOGDescriptor((72, 48), (16, 16), (8, 8), (8, 8), 9)
    hog_list = []
    for i in range(ima_list.shape[0]):
        pic = ima_list[i]
        pic = np.reshape(pic, [40, 60])
        pic = pic.astype(np.uint8)
        pic = cv.resize(pic, (72, 48))
        hog_list.append(hog.compute(pic))
    print('HOG结束计算')
    return hog_list


def cvt_label_form(ima_label):
    new_label = []
    for i in range(ima_label.shape[0]):
        if ima_label[i][0] == 1:
            new_label.append(1)
        else:
            new_label.append(-1)
    return new_label


# 获取svm参数
def get_svm_detector(svm):
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)


if_train = False

if if_train:
    ima_list = np.load('data/train_data.npy')
    ima_label = np.load('data/train_label.npy')
    gradient_list = compute_HOG(ima_list)
    labels = cvt_label_form(ima_label)

    svm = cv.ml.SVM_create()
    svm.setCoef0(0)
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
    svm.setC(0.01)  # From paper, soft classifier
    svm.setType(cv.ml.SVM_EPS_SVR)  # C_SVC # EPSILON_SVR # may be also NU_SVR # do regression task
    print('开始训练')
    svm.train(np.array(gradient_list), cv.ml.ROW_SAMPLE, np.array(labels))
    print('训练完成')
    # 保存结果
    hog = cv.HOGDescriptor((72, 48), (16, 16), (8, 8), (8, 8), 9)
    hog.setSVMDetector(get_svm_detector(svm))
    hog.save('myHOG.bin')
else:
    hog = cv.HOGDescriptor((72, 48), (16, 16), (8, 8), (8, 8), 9)
    hog.load('myHOG.bin')
    for i in range(20, 130):
        name = 'D:\\show_pic\\car_detection\\' + str(i) + '.jpg'
        ima = cv.imread(name, 0)
        rect, wei = hog.detectMultiScale(ima, hitThreshold=0.5, winStride=(4, 4), padding=(8, 8), scale=1.1)
        print(wei)
        for i, (x, y, w, h) in enumerate(rect):
            cv.putText(ima, str(wei[i]), (x, y), cv.FONT_HERSHEY_PLAIN, 1, 255)
            cv.rectangle(ima, (x, y), (x + w, y + h), 255, 2)
        cv.imshow('Detection', ima)
        cv.waitKey(1000)
