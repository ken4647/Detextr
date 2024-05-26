import torch
import numpy as np
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from diffusers.utils import load_image, make_image_grid
import PIL
import cv2



def diffuser_generate_image(init_image_path: str, control_image_path: str, mask_arr_path: str, prompt=["some texts in same style"], negative_prompt= ["low quality, Abrupt"], guidance_scale:float=7.5, strength:float=30, num_inference_steps:int=20) -> np.ndarray:
    # prepare control image
    def make_inpaint_condition(init_image, mask_arr):
        init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0

        # print(mask_arr.shape)
        assert init_image.shape[0:1] == mask_arr.shape[0:1], "image and image_mask must have the same image size"
        init_image[mask_arr > 0.5] = -1.0  # set as masked pixel
        init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
        init_image = torch.from_numpy(init_image)
        return init_image
    
    # load ControlNet
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16, variant="fp16")

    # pass ControlNet to the pipeline
    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16"
    )
    
    # enable CPU offload and memory-efficient attention for faster inference
    pipeline.enable_model_cpu_offload()
    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    pipeline.enable_xformers_memory_efficient_attention()

    # load base and mask image
    init_image = load_image(init_image_path)
    mask_image = load_image(mask_arr_path)
    control_image = np.array(load_image(control_image_path))*255
    control_image = torch.from_numpy(np.expand_dims(control_image, 0).transpose(0, 3, 1, 2))
    
    print(f"Prompt: {prompt}")
    print(f"Negative Prompt: {negative_prompt}")
    print(f"Init image shape: {type(init_image)} {init_image.size}")
    print(f"Mask image shape: {type(mask_image)} {mask_image.size}")
    print(f"Control image shape:{type(control_image)} {control_image.shape}")

    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image, control_image=control_image,guidance_scale=7.5, strength=50, num_inference_steps=20).images[0]
    make_image_grid([init_image, mask_image, PIL.Image.fromarray(np.uint8(control_image[0][0])).convert('RGB'), image], rows=2, cols=2)
    # print(f"Output image shape: {type(image)} {image.size}")
    # cv2.imwrite("output.png", np.array(image))
    return np.array(image)
