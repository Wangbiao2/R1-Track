# Copyright (c) 2025 Biao Wang. All rights reserved.
# Use of this source code is governed by the MIT license that can be
# found in the LICENSE file.

"""A simple R1-Track infer script for GOT-10k dataset evaluation."""

import os
import sys
import re
import json
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2 as cv
import numpy as np
import requests
from tqdm import tqdm
import gc

from crop_image import sample_target, map_bbox_back


class R1TRACK:
    """R1-Track v0.0 only support eval GOT10k dataset."""

    def __init__(self, 
                 hostname="1.1.1.1",
                 port=8888,
                 model_name="R1-Track", 
                 is_think_model=True,
                 max_workers=8,
                 mllm_img_size=336,
                 template_area_factor=4.0,
                 search_area_factor=4.0,
                 temperature=0,
                 top_p=0.5,
                 top_k=20,
                 max_tokens=512,
                 dataset_path="got10k_test",
                 result_path="got10k_result",
                 ):

        """Initialize R1Track tracker.

        Args:
            hostname: API server hostname.
            port: API server port.
            model_name: Name of the tracking model.
            is_think_model: Whether to use thinking model format.
            max_workers: Thread pool size for parallel processing.
            mllm_img_size: Image size for MLLM processing.
            template_area_factor: Scaling factor for template area.
            search_area_factor: Scaling factor for search area.
            temperature: Sampling temperature for model inference.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            max_tokens: Maximum output tokens from the model.
            dataset_path: Path to GOT-10k dataset.
            result_path: Path to store tracking results.
        """

        self.url = f"http://{hostname}:{port}/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        self.model_name = model_name
        self.hostname = hostname
        self.port = port
        self.max_workers = max_workers

        self.is_think_model = is_think_model
        if self.is_think_model:
            self.format_prompt = """You FIRST think about the reasoning process as an internal monologue and then provide the final answer. \n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in <answer> </answer> tags."""
        else:
            self.format_prompt = None
        self.prompt = """Please identify the target specified by the bounding box <BBOXFLAG> in the first image and locate it in the second image. Return the coordinates in [x_min,y_min,x_max,y_max] format."""
        if is_think_model:
            self.prompt = " ".join((self.format_prompt.strip(), self.prompt))

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens

        self.mllm_img_size = mllm_img_size
        self.template_area_factor = template_area_factor
        self.search_area_factor = search_area_factor

        self.dataset_path = dataset_path
        self.result_path = result_path

        video_names = np.loadtxt(os.path.join(dataset_path, 'list.txt'), dtype=str)
        self.video_names = video_names.tolist()
    
    def _clear_cache(self):
        """Release cached data to free memory."""
        del self.bbox_all
        del self.time_all
        del self.xyxy_last
        del self.im1_crop
        del self.bbox1_crop
        del self.last_bbox


    def track_all_videos(self):
        """Process all videos using thread pool executor."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for video_name in self.video_names:
                tracker = R1TRACK(
                    hostname=self.hostname,
                    port=self.port,
                    model_name=self.model_name,
                    is_think_model=self.is_think_model,
                    mllm_img_size=self.mllm_img_size,
                    template_area_factor=self.template_area_factor,
                    search_area_factor=self.search_area_factor,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    max_tokens=self.max_tokens,
                    dataset_path=self.dataset_path,
                    result_path=self.result_path
                )
                futures.append(executor.submit(tracker.process_single_video, video_name))
            
            for future in tqdm(as_completed(futures), total=len(self.video_names)):
                try:
                    future.result()
                except Exception as e:
                    print(f"Video processing failed: {e}")
            executor.shutdown(wait=True)

    def process_single_video(self, video_name):
        """Process single video and save tracking results.

        Args:
            video_name: Name of the video sequence to process.
        """
        try:
            self.initialize(video_name)
            save_dir = os.path.join(self.result_path, video_name)
            os.makedirs(save_dir, exist_ok=True)
            bbox_save, time_save = self.track()
            
            # Save tracking results
            with open(os.path.join(save_dir, f"{video_name}_001.txt"), 'w') as f:
                f.writelines([','.join(map(str, row)) + '\n' for row in bbox_save])
            with open(os.path.join(save_dir, f"{video_name}_time.txt"), 'w') as f:
                f.writelines([f"{t}\n" for t in time_save])

        finally:
            self._clear_cache()
            gc.collect()
            cv.destroyAllWindows()

    def get_response_from_api(self, im_1, im_2, bbox_1):
        """Send images to API and parse response.

        Args:
            image_1: First image (template).
            image_2: Second image (search area).
            bbox_1: Bounding box coordinates in [x_min, y_min, x_max, y_max] format.

        Returns:
            Parsed bounding box coordinates.
        """
        im_1_base64 = self.img2base64(im_1)
        im_2_base64 = self.img2base64(im_2)

        self.prompt = self.prompt.replace("<BBOXFLAG>", "[{},{},{},{}]".format(*bbox_1))

        data = {
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": im_1_base64
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": im_2_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": self.prompt,
                        }
                    ]
                }
            ]
        }

        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
            response.raise_for_status()
            response_data = response.json()
            print(response_data['choices'][0]['message']['content'])
            response_real = response_data['choices'][0]['message']['content']
            if self.is_think_model:
                after_parse = self.check_and_extract(response_real)
                parse_bbox = json.loads(after_parse) if after_parse else self.xyxy_last
            else:
                parse_bbox = json.loads(response_real)
            if len(parse_bbox) != 4:
                return self.xyxy_last
            return parse_bbox
        except:
            return self.xyxy_last

    def get_sorted_image_paths(self, video_path):
        """Sort image paths in video directory numerically.

        Args:
            video_path: Path to video sequence directory.

        Returns:
            Sorted list of absolute image paths.
        """
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
            print(f"Error processing image paths: {str(e)}")
            return []

    def read_1st_frame_groundtruth(self, file_path):
        """Read groundtruth bounding box from file.

        Args:
            file_path: Path to groundtruth.txt file.

        Returns:
            Bounding box coordinates as list of floats.
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                return [float(num) for num in content.split(',')]
        except Exception as e:
            return []

    def img2base64(self, img):
        """Convert OpenCV image to base64 string.

        Args:
            image: OpenCV image array.

        Returns:
            Base64 encoded JPEG image string.
        """
        try:
            _, buffer = cv.imencode(".jpg", img)
            jpg_base64 = base64.b64encode(buffer).decode("utf-8")
            jpg_base64_with_header = f"data:image/jpeg;base64,{jpg_base64}" 
            return jpg_base64_with_header
        finally:
            del img
            gc.collect()

    def check_and_extract(self, s):
        """Extract answer from thinking model response.

        Args:
            response: Raw model response string.

        Returns:
            Extracted answer content or empty string.
        """
        pattern = r'^<think>.*</think>.*<answer>(.*?)</answer>$'
        match = re.fullmatch(pattern, s, flags=re.DOTALL)
        if match:
            return match.group(1)
        else:
            return 0.0

    def initialize(self, video_name="GOT-10k_Test_000001"):
        """Initialize tracking sequence.

        Args:
            video_name: Name of the video sequence.
        """
        video_path = os.path.join(self.dataset_path, video_name)
        gt_path = os.path.join(video_path, "groundtruth.txt")

        self.image_paths = self.get_sorted_image_paths(video_path)
        video_length = len(self.image_paths)
        print(f"Video {video_name} has {video_length} images.\n")

        bbox1 = self.read_1st_frame_groundtruth(gt_path)
        self.last_bbox = bbox1 # [x,y,w,h]

        self.bbox_all = []
        self.bbox_all.append(self.last_bbox)

        self.time_all = []
        self.time_all.append(0.0)

        im1 = cv.imread(self.image_paths[0])
        self.im1_crop, _, self.bbox1_crop, _ = sample_target(im1, bbox1, self.template_area_factor, self.mllm_img_size)
        self.xyxy_last = self.bbox1_crop

    def track(self):
        for idx, cur_frame_path in enumerate(tqdm(self.image_paths[1:])):
            start_time = time.time()
            im2 = cv.imread(cur_frame_path)
            h, w, _ = im2.shape
            
            im2_crop, resize_factor, _, cache_x1y1 = sample_target(
                im2, self.last_bbox, self.search_area_factor, self.mllm_img_size
            )
            
            bbox2_xyxy = self.get_response_from_api(self.im1_crop, im2_crop, list(map(int, self.bbox1_crop)))
            
            valid_bbox = [
                max(0, min(bbox2_xyxy[0], self.mllm_img_size)),
                max(0, min(bbox2_xyxy[1], self.mllm_img_size)),
                max(0, min(bbox2_xyxy[2], self.mllm_img_size)),
                max(0, min(bbox2_xyxy[3], self.mllm_img_size))
            ]
            if valid_bbox[0] >= valid_bbox[2] or valid_bbox[1] >= valid_bbox[3]:
                valid_bbox = self.xyxy_last
                fail_count += 1
            else:
                fail_count = 0
            
            last_bbox_new = map_bbox_back(resize_factor, cache_x1y1, valid_bbox)
            
            x_min = max(0, last_bbox_new[0])
            y_min = max(0, last_bbox_new[1])
            box_w = max(1, min(last_bbox_new[2], w - x_min))
            box_h = max(1, min(last_bbox_new[3], h - y_min))
            if x_min == 0 or y_min == 0 or box_w == 1 or box_h == 1 or box_w == w - x_min or box_h == h - y_min:
                self.bbox_all.append(self.last_bbox)
                self.time_all.append(time.time() - start_time)
                continue                
            else:
                self.last_bbox = last_bbox_new

            self.xyxy_last = valid_bbox
            
            self.bbox_all.append(self.last_bbox)
            self.time_all.append(time.time() - start_time)
        
        return self.bbox_all, self.time_all


if __name__ == "__main__":
    r1track = R1TRACK(
                 hostname="xx.xx.xxx.xx",
                 port=8888,
                 model_name="R1-Track", 
                 is_think_model=True,
                 max_workers=8,
                 mllm_img_size=336,
                 template_area_factor=4.0,
                 search_area_factor=4.0,
                 temperature=0,
                 top_p=0.5,
                 top_k=20,
                 max_tokens=512,
                 dataset_path="xxx",
                 result_path="xxx",)
    r1track.track_all_videos()