import os
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq
import pyarrow as pa
from datasets import Dataset, Features, Sequence, Value


def read_bbox(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()


def generate_data(base_path):
    for folder in os.listdir(base_path):
        if "._" in folder:
            continue
        if ".DS_Store" in folder:
            continue
        folder_path = os.path.join(base_path, folder)
        try:
            with open(os.path.join(folder_path, 'image_1.jpg'), 'rb') as f:
                img1_bytes = f.read()
            with open(os.path.join(folder_path, 'image_2.jpg'), 'rb') as f:
                img2_bytes = f.read()

            bbox1 = read_bbox(os.path.join(folder_path, 'bbox_1.txt'))
            bbox2 = read_bbox(os.path.join(folder_path, 'bbox_2.txt'))

            problem = f"""<image> <image>Given two images, you need to:
1. Analyze and Identify the target object marked by bounding box [{bbox1}] in <image_1>;
2. Re-locate this target in <image_2>;
3. Return [x_min, y_min, x_max, y_max] coordinates of the target in <image_2>."""
            
            images = [
                {"bytes": img1_bytes, "path": "image_1.jpg"},
                {"bytes": img2_bytes, "path": "image_2.jpg"},
            ]
            
            yield {
                "images": images,
                "problem": problem,
                "answer": f"[{bbox2}]"
            }
        except Exception as e:
            print(f"\nError processing {folder}: {str(e)}")
            continue


def main():
    base_path = "./R1-Track-336-v0"
    output_dir = "./huggingface-data/train"
    os.makedirs(output_dir, exist_ok=True)

    dataset = Dataset.from_generator(
        generate_data,
        gen_kwargs={"base_path": base_path}
    )
    
    shuffled_ds = dataset.shuffle(seed=42)
    shuffled_ds.to_parquet(os.path.join(output_dir, "R1-Track-336-v0.parquet"))

if __name__ == "__main__":
    main()