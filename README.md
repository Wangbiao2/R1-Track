# R1-Track
- This Is A Simple Single Object Tracking Repo Based on Qwen2.5-VL MLLM with SFT and ‌RFT.
- Visual (Single) Object Tracking aims to continuously localize and estimate the scale of a target in subsequent video frames, given only its initial state in the first frame. This task can be simplified to template matching between image pairs, with traditional trackers predominantly employing explicit classification-regression modeling through Correlation Filters, Siamese networks, and Vision Transformers (ViT). Leveraging advancements in Multi-Modal Large Language Models (MLLMs) such as Qwen2.5-VL and their robust grounding capabilities, we explore adopting MLLMs for end-to-end tracking tasks, eliminating the need for fragmented subtask modeling.
- The checkpoints, training pipeline, inference scripts and data will be available before April 30, 2025.


## Prompt
- SFT
```python
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": xxx,
            },
            {
                "type": "image",
                "image": xxx,
            },
            {"type": "text", "text": "Please identify the target specified by the bounding box [241,66,329,154] in the first image and locate it in the second image. \n Return the coordinates in [x_min,y_min,x_max,y_max] format."},
        ],
    }
```

- RFT
```python
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": xxx,
            },
            {
                "type": "image",
                "image": xxx,
            },
            {"type": "text", "text": "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. \n The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in <answer> </answer> tags. \n Please identify the target specified by the bounding box [241,66,329,154] in the first image and locate it in the second image. \n Return the coordinates in [x_min,y_min,x_max,y_max] format."},
        ],
    }
```

# Reference Repos
- Base Model: We use [Qwen2.5-Vl-Instruct-3B](https://github.com/QwenLM/Qwen2.5-VL) as our base model.
- SFT: We perform supervised fine-tuning based on [LLama-Factory](https://github.com/hiyouga/LLaMA-Factory).
- RFT: We conduct reinforcement learning fine-tuning using [EasyR1](https://github.com/hiyouga/EasyR1).
- Track: [PyTracking](https://github.com/visionml/pytracking).


## Data
- R1-Track-5k dataset is available at https://huggingface.co/datasets/WangBiao/R1-Track-5k. 
Note that this dataset was randomly sampled from [Got10k](http://got-10k.aitestunion.com/) and has not undergone manual review. Some image pairs are of relatively low quality, but we will address and improve this issue in the future.
<img width="500" alt="image" src="https://github.com/user-attachments/assets/25afecd3-16b9-4a02-a816-eb2a8bf63ba4" />


## Quick Start
A detailed user guide will be launched in the near future.


## Timeline
- [2025/04/02] **We released [R1-Track-5K] dataset. (https://huggingface.co/datasets/WangBiao/R1-Track-5k)**!
- [2025/04/20] **We released [R1-Track-SFT] model. (https://huggingface.co/WangBiao/R1-Track-SFT)**!
