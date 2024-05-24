from utils.create_txt import create_txt
from image_process import *
from FETGAN import run_fetgan
from diffuser_opt import *

DEFEALT_SIZE = (320, 320)
FET_OUTPUT_PATH = "outputs/transfer/test_latest/images/1_transfered.png"

# step1: input position of text information
# step2: crop text from image
# step3: padding image with average of edge background
# step4: run FETGAN to transfer text into image
# step5: add edge oftransfered image into original image's ege
# step6: run diffusion inpainting to fill the missing area
if __name__ == '__main__':
    orginal_image = cv.imread(sys.argv[1])
    
    txt_image,pos = crop_detect(orginal_image)
    ref_image = padding_into(txt_image, DEFEALT_SIZE)
    
    create_txt(input("Enter the text you want to change into: "))
    run_fetgan()
    
    tansfered_image = cv.imread(FET_OUTPUT_PATH)
    edge = edge_canny(ref_image)
    edge_addinto(orginal_image, tansfered_image, pos)
    
    run_diffusion_inpainting(orginal_image)
    
    pass
