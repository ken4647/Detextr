import cv2 as cv
import numpy as np
import sys

def edge_canny(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)  # 高斯模糊
    gray = cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)  # 灰路图像
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)  # xGrodient
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)  # yGrodient
    edge_output = cv.Canny(xgrad, ygrad, 200, 220)  # edge
    return edge_output

    # #  彩色边缘
    # dst = cv.bitwise_and(image, image, mask=edge_output)


src=cv.imread(sys.argv[1])
cv.imwrite("process_gray.png", edge_canny(src))