import cv2
import math
import random
import os
import numpy as np
from tqdm import tqdm


def sample_target(im, target_bb, search_area_factor, output_sz, radio):
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb

    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    if crop_sz < 1:
        raise ValueError("Bounding box too small")

    x_center = x + 0.5 * w
    y_center = y + 0.5 * h

    dx = random.uniform(-radio * crop_sz, radio * crop_sz)
    dy = random.uniform(-radio * crop_sz, radio * crop_sz)
    
    x1 = round(x_center + dx - crop_sz * 0.5)
    y1 = round(y_center + dy - crop_sz * 0.5)
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    y1_pad = max(0, -y1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, 
                                    x1_pad, x2_pad, 
                                    cv2.BORDER_CONSTANT)

    resize_factor = output_sz / crop_sz if output_sz else 1.0
    if output_sz:
        im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
    
    x_in_crop = (x - x1) * resize_factor
    y_in_crop = (y - y1) * resize_factor
    w_resized = w * resize_factor
    h_resized = h * resize_factor

    target_bb_in_crop = [
        x_in_crop,
        y_in_crop,
        x_in_crop + w_resized,
        y_in_crop + h_resized
    ]

    return im_crop_padded, list(map(int, target_bb_in_crop))

def process_image_sequence(image_paths, bboxes, output_sz, save_path, offset_radio, factor_min, factor_max, num_images):
    try:
        selected = random.sample(list(zip(image_paths, bboxes)), num_images)

        for i, (img_path, bbox) in enumerate(selected, 1):
            im = cv2.imread(img_path)
            
            search_area_factor = random.uniform(factor_min, factor_max)

            cropped_img, bbox_in_crop = sample_target(
                im, bbox, search_area_factor, output_sz, offset_radio
            )
            
            cv2.imwrite(os.path.join(save_path, f"image_{i}.jpg"), cropped_img)
            
            with open(os.path.join(save_path, f"bbox_{i}.txt"), "w") as f:
                f.write(",".join(map(str, bbox_in_crop)))
    except:
        pass

def get_sorted_image_paths(video_path):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    try:
        all_files = os.listdir(video_path)
        image_files = [f for f in all_files 
                    if os.path.splitext(f.lower())[1] in valid_extensions]
        image_files_sorted = sorted(image_files, 
                                key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
        absolute_paths = [os.path.abspath(os.path.join(video_path, f)) 
                        for f in image_files_sorted]
        return absolute_paths
    except Exception as e:
        print(f"Error encountered during image path processing: {str(e)}")
        return []
    

def read_txt_to_2d_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        return [[float(num) for num in line.strip().split(',')] for line in lines]


if __name__ == "__main__":

    resolution = 336
    version = "v0"
    offset_radio = 0.2
    factor_min = 2
    factor_max = 8
    num_images = 2

    dataset_path = "./got10k/train_data/"
    video_names = np.loadtxt(os.path.join(dataset_path, 'list.txt'), dtype=str)
    video_names = video_names.tolist()

    base_path = "./R1-Track-data/R1-Track-" + str(resolution) + '-' + version + '/'
    max_name_len = 6

    for idx, video_name in tqdm(enumerate(video_names)):
        cur_name_len = len(str(idx+1))
        save_path = base_path + 'pair_' + '0'*(max_name_len - cur_name_len) + str(idx+1) + '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        video_path = os.path.join(dataset_path, video_name)
        image_paths = get_sorted_image_paths(video_path)
        gt_path = os.path.join(video_path, "groundtruth.txt")
        bboxes = read_txt_to_2d_list(gt_path)
        process_image_sequence(image_paths, bboxes, resolution, save_path, offset_radio, factor_min, factor_max, num_images)

