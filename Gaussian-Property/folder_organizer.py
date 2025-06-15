import os
import argparse
from PIL import Image
from rembg import remove
from utils.sam_utils import resize_image
from tqdm import tqdm
import re

def numerical_sort_key(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else -1

def process_images(remove_bg):
    for obj_name in os.listdir(BASE_PATH):
        obj_path = os.path.join(BASE_PATH, obj_name)
        if not os.path.isdir(obj_path):
            continue
        image_dir = os.path.join(SAVE_BASE_PATH, obj_name, 'images')
        os.makedirs(image_dir, exist_ok=True)
        images = sorted(os.listdir(obj_path), key=numerical_sort_key)
        for idx, image_name in tqdm(enumerate(images), total=len(images), desc=f'Processing {obj_name}'):
            image_path = os.path.join(obj_path, image_name)
            with Image.open(image_path) as img_pil:
                if remove_bg:
                    img_pil = remove(img_pil)
                img_pil = resize_image(img_pil, 1280)
                mask_save_path = os.path.join(image_dir, f'{idx+1:03d}.png')
                img_pil.save(mask_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images with optional background removal.")
    parser.add_argument('--no-remove', action='store_false', dest='remove', help="Set this flag to skip background removal.")
    parser.add_argument('--folder_path', type=str, default='gp_cases', help="Path to the folder containing the images.")
    
    args = parser.parse_args()
    # Constants for directory paths
    BASE_PATH = args.folder_path
    SAVE_BASE_PATH = args.folder_path + '_dirs'

    process_images(args.remove)