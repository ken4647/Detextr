import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys

FFT_PTAH = "./fft/NotoSansSC-Black.ttf"
OUTPUT_DIR = "./image_outputs/plain_images"

def create_image_from_text(text=None, 
                           size=[128, 128], 
                           font_size=70):
    '''Create an image with an input text
        Parameter:
            word : string
            size : tuple
            font_size: int 
    '''
    
    height, width = size
    background = np.zeros((height, width), dtype="uint8") + 255
    image_pil = Image.fromarray(background)
    
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(FFT_PTAH, font_size)
    
    # 获取文本边界框
    bbox = draw.textbbox((0, 0), text, font=font)
    
    print(f"bbox {bbox}")
    
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    text_x = (width - text_width) / 2
    text_y = (height - text_height) / 2 - bbox[1]

    draw.text((text_x, text_y), text, fill=0, font=font)
    
    # 转换为 numpy 数组
    image = np.array(image_pil)

    return image
    
