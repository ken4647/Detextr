import cv2
import numpy as np
import sys
import math

def edge_canny(image=None):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)  # 高斯模糊
    gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)  # 灰路图像
    xgrad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)  # xGrodient
    ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)  # yGrodient
    edge_output = cv2.Canny(xgrad, ygrad, 200, 220)  # edge
    return edge_output



def center_crop_resize(image=None, target_width=None, target_height=None):
    current_height, current_width = image.shape[:2]
    scale = max(target_width / current_width, target_height / current_height)
    new_width = int(current_width * scale)
    new_height = int(current_height * scale)
    resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    start_x = (new_width - target_width) // 2
    start_y = (new_height - target_height) // 2
    cropped_img = resized_img[start_y:start_y + target_height, start_x:start_x + target_width]
    
    return cropped_img


def crop_image(image=None, position=None, padding=False, crop_size=(128, 128)):
    """
    Crop image and optionally pad it, then resize while keeping content centered.
    
    Parameters:
    - image: np.ndarray, input image
    - position: tuple of tuples, [(x1, y1), (x2, y2)]
      (x1, y1): top-left
      (x2, y2): bottom-right
    - padding: bool, if True, pad the cropped image with the average of edge background
    - crop_size: tuple, desired size of the output cropped image (width, height)
    
    Returns:
    - np.ndarray, cropped (and possibly padded) image
    """
    if image is None or position is None:
        raise ValueError("Image and position must be provided")

    (x1, y1), (x2, y2) = position
    cropped_img = image[y1:y2, x1:x2]
    
    if padding:
        current_height = abs(y1 - y2)
        current_width = abs(x1 - x2)
        top_padding = max((192 - current_height) // 2, 0)
        bottom_padding = max(192 - current_height - top_padding, 0)
        left_padding = max((192 - current_width) // 2, 0)
        right_padding = max(192 - current_width - left_padding, 0)
        
        padded_img = cv2.copyMakeBorder(
            cropped_img,
            top_padding, bottom_padding, left_padding, right_padding,
            cv2.BORDER_REPLICATE
        )
        cropped_img = padded_img
    
    cropped_img = center_crop_resize(image=cropped_img,target_width=crop_size[0], target_height=crop_size[1])
    
    return cropped_img

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
