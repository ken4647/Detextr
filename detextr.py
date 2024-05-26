import os
from utils.create_txt import create_txt
from image_process import *
from FETGAN import run_fetgan
from diffuser_opt import *

DEFEALT_SIZE = (320, 320)

if not os.path.exists("outputs"):
    os.makedirs('outputs', exist_ok=True)

FET_OUTPUT_PATH = "outputs/transfer/test_latest/images/1_transfered.png"
DIFFUSION_RESULT_PATH = "outputs/diffusion/images/1_diffused.png"

# step1: input position of text information
# step2: crop text from image
# step3: padding image with average of edge background
# step4: run FETGAN to transfer text into image
# step5: add edge oftransfered image into original image's ege
# step6: run diffusion inpainting to fill the missing area
if __name__ == '__main__':
    orginal_image = cv.imread(sys.argv[1])
    
    txt_image,rect = crop_detect(orginal_image)
    ref_image = padding_into(txt_image, DEFEALT_SIZE)
    
    create_txt(input("Enter the text you want to change into: "))
    run_fetgan()
    
    tansfered_image = cv.imread(FET_OUTPUT_PATH)
    tansfered_edge = edge_canny(tansfered_image)
    
    local_img, local_rect = rgb_addinto_and_crop(orginal_image, tansfered_image, rect)
    edge_img = edge_addinto(local_img, tansfered_edge, local_rect)
    mask_img = mask_get(local_img, local_rect)
    
    run_diffusion_inpainting(orginal_image)
    
    recover_into(orginal_image, DIFFUSION_RESULT_PATH)
    
    pass
