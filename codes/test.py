import sys
sys.path.extend(['..'])

import os
from utils.create_image_from_text import *
from codes.process_image import *
from codes.run_fetgan import run_fetgan
from diffuser_opt import *
from utils.util import *

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

transfered_image_path = '../image_repos/transferred_crop_images/TextEffects/test_latest/images/1_transfered.png'
transfered_image = cv2.imread(transfered_image_path)
edge_transfered_image = edge_canny(transfered_image)

resized_transfer_image, tesized_edge_image = crop_to_edges(image=transfered_image, edge_image=edge_transfered_image)

cv2.imwrite('./test1.png', resized_transfer_image)
cv2.imwrite('./test2.png', tesized_edge_image)