import cv2
import numpy as np
import sys
import math
from PIL import Image


def edge_canny(image=None):
    blurred = cv2.GaussianBlur(image, (3, 3), 0) 
    gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)  
    xgrad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0) 
    ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    edge_output = cv2.Canny(xgrad, ygrad, 200, 220)  
    return edge_output

def crop_to_edges(image=None, edge_image=None):
    """
    Crop the image to the bounding box of the edges detected in the edge_image,
    with a 5-pixel padding.

    Parameters:
    - image: The original input image.
    - edge_image: The edge-detected image.

    Returns:
    - cropped_image: The cropped image containing only the region with edges.
    """
    if edge_image is None:
        raise ValueError("image and edge_image parameters cannot be None")
    
    # Find the bounding box of the non-zero regions in the edge image
    coords = cv2.findNonZero(edge_image)  # Find all non-zero points (i.e., edge points)
    x, y, w, h = cv2.boundingRect(coords)  # Find the bounding box of these points

    # Add 5-pixel padding and ensure coordinates are within image boundaries
    padded_x1 = max(x - 10, 0)
    padded_y1 = max(y - 10, 0)
    padded_x2 = min(x + w + 10, image.shape[1])
    padded_y2 = min(y + h + 10, image.shape[0])
    
    cropped_image = None
    if image is not None:
        cropped_image = image[padded_y1:padded_y2, padded_x1:padded_x2]
    cropped_edge_image = edge_image[padded_y1:padded_y2, padded_x1:padded_x2]
    
    return cropped_image, cropped_edge_image


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

def paste_image(image1=None, image2=None, position=None, resize_tgt_image=True):
    """
    Paste image2 onto image1 at the specified position.
    
    Parameters:
    - image1: Background image
    - image2: Image to paste
    - position: Coordinates [(x1, y1), (x2, y2)] where
        (x1, y1): Top-left corner
        (x2, y2): Bottom-right corner
    - resize_tgt_image: Whether to resize image2 to fit within the specified region in image1
    
    Returns:
    - Combined image
    """
    if image1 is None or image2 is None or position is None:
        raise ValueError("image1, image2, and position parameters cannot be None")
    
    (x1, y1), (x2, y2) = position
    h, w = image2.shape[:2]

    # Calculate the dimensions of the region in image1 where image2 will be pasted
    region_height = y2 - y1
    region_width = x2 - x1
    
    # Resize image2 to fit within the specified region
    if resize_tgt_image:
        image2 = cv2.resize(image2, (region_width, region_height))
        h, w = image2.shape[:2]
    
    # Ensure the resized image2 fits within the specified region
    if h > region_height or w > region_width:
        raise ValueError("Resized image2 exceeds the specified region's boundaries")

    # Paste the resized image2 onto image1 at the specified position
    image1[y1:y1+h, x1:x1+w] = image2
    
    return image1

def blackout_image(image=None, position=None):
    # Convert image to RGB and normalize to [0, 1]
    np_image = np.array(image).astype(np.float32) / 255.0
    
    # Get the mask position
    (x1, y1), (x2, y2) = position
    
    # Set the specified region to -1
    np_image[y1 : y2, x1 : x2] = -1.0
    np_image = np.expand_dims(np_image, 0).transpose(0, 3, 1, 2)
    return np_image   


def convert_to_image(np_image=None):
    """
    Convert the processed NumPy array back to a PIL.Image for display or saving.

    Parameters:
    - np_image: np.ndarray, input image in [0, 1] range with -1 in the masked region
    
    Returns:
    - PIL.Image, the resulting image
    """
    # Remove the added dimension and transpose back to original shape
    np_image = np_image.transpose(0, 2, 3, 1).squeeze(0)
    
    # Reverse normalization and set masked region to black (0)
    np_image = (np_image * 255.0).astype(np.uint8)
    np_image[np_image == -255] = 0
    
    return np_image


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
        top_padding = max((224 - current_height) // 2, 0)
        bottom_padding = max(224 - current_height - top_padding, 0)
        left_padding = max((224 - current_width) // 2, 0)
        right_padding = max(224 - current_width - left_padding, 0)
        
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
