'''This is the main .py
'''

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

'''
    sys.argv
        [1] image id
        [2] crop id (position)
        [3] type a new letter/character
        
    e.g. python detextr 1 1 永
    e.g. python detextr 1 2 6
    e.g. python detextr 1 3 口
'''
def get_options():
    opt = get_option('../configs/detextr.yaml')
    opt['image_id'] = sys.argv[1]
    opt['crop_id'] = sys.argv[2]
    opt['rewrite_letter'] = sys.argv[3]
    return opt


# step1: input position of text information
# step2: crop text from image
# step3: padding image with average of edge background
# step4: run FETGAN to transfer text into image
# step5: add edge of transfered image into original image's edge
# step6: run diffusion inpainting to fill the missing area

def main(opt):
    '''Step 1, prepare the original picture and select the letter to be editted
    
        In theory, there is a UI for the user to input the position of the part that is to be edited.
        But currently we prepared the position in advance for ease of implementation.
    '''
    # load the image to be edited 
    image_id = opt['image_id']
    crop_id = opt['crop_id']
   

    # Read image
    image_path = f'../image_repos/original_images/{image_id}/image.png'
    original_image = cv2.imread(image_path)
    
    # Read the box position 
    position_id_path = f'../image_repos/original_images/{image_id}/c{crop_id}.txt'
    with open(position_id_path, 'r') as reader:
        position = [s.split(',') for s in reader.read().strip().split('|')]
    point1, point2 = [[int(x) for x in map(str.strip, sublist)] for sublist in position]
    
    ''' Step 2, crop the letter to be replaced from the original image
                resize it and pad it with the background color
    '''
    cropped_image = crop_image(image=original_image,
                               position=[point1, point2],
                               padding=True,
                               crop_size=[128,128])
    crop_image_dir = f"../image_repos/crop_images/{image_id}/{crop_id}"
    os.makedirs(crop_image_dir, exist_ok=True)
    crop_image_path = f'{crop_image_dir}/{crop_id}.png'
    if os.path.exists(crop_image_path):
        os.remove(crop_image_path)
    cv2.imwrite(crop_image_path, cropped_image)
    
    ''' Step 3, input the letter and generate a new image containing the letter
    '''
    rewrite_letter = opt['rewrite_letter']
    text_image = create_image_from_text(
        text=rewrite_letter,
    )
    
    text_image_dir = f'../image_repos/letter_images/{image_id}'
    os.makedirs(text_image_dir, exist_ok=True)
    text_image_path = f'{text_image_dir}/{crop_id}.png'
    if os.path.exists(text_image_path):
        os.remove(text_image_path)
    cv2.imwrite(text_image_path, text_image)
    
    ''' Step 4, run fet-gan to transfer its style
    '''
    ref_pic_dir = f"../image_repos/crop_images/{image_id}/{crop_id}"
    run_fetgan(src_pic_path=text_image_path,
               ref_pic_path=ref_pic_dir,
               args=opt)

    ''' Step 5, add edge of transfered image into original image's edge
    '''
    # Extract the edge of the original image
    edge_original_image = edge_canny(image=original_image)
    
    # Extract the edge of the transferred image
    transferred_image_path = f'../image_repos/transferred_crop_images/TextEffects/{image_id}/{crop_id}/images/1_transfered.png'
    transferred_image = cv2.imread(transferred_image_path)
    edge_transferred_image = edge_canny(image=transferred_image)
    

    edge_image_dir = f'../image_repos/edge_images/{image_id}'
    os.makedirs(edge_image_dir,exist_ok=True)
    edge_transferred_image_path = f'{edge_image_dir}/{crop_id}.png'
    edge_origianl_image_path = f'{edge_image_dir}/original_edge.png'
    cv2.imwrite(edge_transferred_image_path, edge_transferred_image)
    cv2.imwrite(edge_origianl_image_path, edge_original_image)
    
    input_for_impainting_dir = f'../image_repos/input_for_impainting/{image_id}/{crop_id}'
    os.makedirs(input_for_impainting_dir,exist_ok=True)
    
    # further crop the image to scale the letter
    # i.e. remove the blank 
    resized_transferred_image, resized_edge_image = crop_to_edges(image=transferred_image, edge_image=edge_transferred_image)

    # scale the transferred image and past it to the original image
    pasted_image = paste_image(image1=copy.deepcopy(original_image),
                image2=copy.deepcopy(resized_transferred_image),
                position=[point1, point2], 
                resize_tgt_image=True
                )
    edge_pasted_image = edge_canny(image=pasted_image)
    
    black_out_image_arr = blackout_image(image=copy.deepcopy(pasted_image),
            position=[point1, point2], 
            )   
    
    black_out_image = convert_to_image(black_out_image_arr)

    # Save images 
    cv2.imwrite(f'{input_for_impainting_dir}/pasted_image.png', pasted_image)
    cv2.imwrite(f'{input_for_impainting_dir}/edge_pasted_image.png', edge_pasted_image)
    cv2.imwrite(f'{input_for_impainting_dir}/balck_out_image.png', black_out_image)
    cv2.imwrite(f'{input_for_impainting_dir}/original_image.png', original_image)
    np.save(f'{input_for_impainting_dir}/black_out.npy', black_out_image_arr)

    

if __name__ == '__main__':
    opt = get_options()
    main(opt)
