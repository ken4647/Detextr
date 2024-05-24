import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys

FFT_PTAH = "fft/SourceHanSansSC-Normal.otf"
OUTPUT_PATH = "./inputs/src_imgs/orginal.png"

def create_txt(word):
    # 创建一个空白的图像
    height, width = 128, 128
    truthground = np.zeros((height, width), dtype="uint8") + 255
    # 将OpenCV的图像转换为Pillow的图像
    image_pil = Image.fromarray(truthground)

    # 创建一个可以在Pillow图像上绘图的对象
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype("fft/SourceHanSansSC-Normal.otf", 84)
    w, h = font.getsize(word)
    print(w,h)
    draw.text(((width-w)/2, 0), word, fill=0, font=font, antialias=False)

    # 将Pillow的图像转换回OpenCV的图像
    image = np.array(image_pil)

    # 显示图像
    cv2.imwrite(f'{OUTPUT_PATH}', image)