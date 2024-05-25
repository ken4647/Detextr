'''This is the main .py
'''

import sys
sys.path.extend(['..'])

import os
from utils.create_image_from_text import *
from codes.process_image import *
from FETGAN import run_fetgan
from diffuser_opt import *
from utils.util import *

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


'''
    sys.argv
    [1] image id
    [2] crop id (position)
    [3] letter W
'''
def get_options():
    opt = get_option('../configs/detextr.yaml')
    
    opt['image_id'] = sys.argv[1]
    opt['crop_id'] = sys.argv[2]
    opt['rewrite_letter'] = sys.argv[3]
    
    opt['K'] = 8
    opt['name'] = 'TextEffects' # model's name (folder name): 'TextEffects' / 'Fonts100'
    opt['testresults_dir'] = opt['testresults_dir'] + '/TextEffects' # '/TextEffects' / '/Fonts100'
    opt['isTrain'] = False # train(True) or test(False)
    opt['batch_size'] = 1
    opt['num_threads'] = 0
    opt['display_id'] = -1
    opt['load_size'] = 320
    opt['crop_size'] = 256
    opt['load_iter'] = 30
    opt['typed_word'] = 'Hell'

    opt['testsource'] = None #'./testimgs/TextEffects/FontEffects_sources/github.png' #None # single source image for test
    opt['testsource_dir'] = './testimgs/TextEffects/TextEffects_sources/' # source image folder for test './testimgs/Fonts100/Fonts100_sources/', './testimgs/TextEffects/TextEffects_sources/'

    opt['testrefs'] = None #'./testimgs/TextEffects/FontEffects_refs/derby' #None # single ref images folder for test
    opt['testrefs_dir'] = './testimgs/TextEffects/TextEffects_refs' # multiple reference images folders for test './testimgs/Fonts100/Fonts100_refs', './testimgs/TextEffects/TextEffects_refs'

    return opt


# step1: input position of text information
# step2: crop text from image
# step3: padding image with average of edge background
# step4: run FETGAN to transfer text into image
# step5: add edge oftransfered image into original image's ege
# step6: run diffusion inpainting to fill the missing area


def main(opt):
    # load the image to be edited 
    
    image_id = opt['image_id']
    crop_id = opt['crop_id']
    
    image_path = f'../image_repos/original_images/{image_id}/image.png'

    # Read image
    orginal_image = cv2.imread(image_path)
    
    '''Step 0, prepare the original picture and select the letter to be cropped.
    
        In theory, there is a UI for the user to input the position of the part that is to be edited.
        But currently we prepared the position in advance for ease of implementation.
    '''
    position_id_path = f'../image_repos/original_images/{image_id}/c{crop_id}.txt'
    with open(position_id_path, 'r') as reader:
        position = [s.split(',') for s in reader.read().strip().split('|')]
    point1, point2 = [[int(x) for x in map(str.strip, sublist)] for sublist in position]
    
    cropped_image = crop_image(image=orginal_image,
                               position=[point1, point2],
                               padding=True,
                               crop_size=[128,128])
    
    crop_image_path = f'../image_repos/crop_images/{image_id}_{crop_id}_cropped_image.png'
    if os.path.exists(crop_image_path):
        os.remove(crop_image_path)
    cv2.imwrite(crop_image_path, cropped_image)
    
    
    ''' Step 1, input the letter and transfer style
    '''
    rewrite_letter = opt['rewrite_letter']



if __name__ == '__main__':
    opt = get_options()
    main(opt)
