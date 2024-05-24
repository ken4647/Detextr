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


# step1: input position of text information
# step2: crop text from image
# step3: padding image with average of edge background
def crop_detect(image):
    pass

# input position of text information by ui (use opencv)
def crop_detect_ui(image):
    pass

# padding text into suitable size for FET_GAN
def padding_into(image, pad_size):
    pass

# resize image to suitable size for diffusion
def edge_addinto(edge_image1, edge_image2, pos):
    pass

def rgb_addinto_and_crop(rgb_image1, rgb_image2, pos):
    pass

def combine_image(rgb_image, local_image):
    pass

def mask_get(size, rect):
    pass

def recover_into(image, rect):
    pass
