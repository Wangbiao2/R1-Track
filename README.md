# R1-Track: Direct Application of MLLMs to Visual Object Tracking via Reinforcement Learning

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

- RL
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
- Object Track: [PyTracking](https://github.com/visionml/pytracking).
- Inference: [vllm](https://github.com/vllm-project/vllm).
- Data: We utilized a portion of the [GOT10k-train](http://got-10k.aitestunion.com/) dataset to assemble our training set and conducted one-shot testing exclusively on [GOT10k-test](http://got-10k.aitestunion.com/).


## Data
- R1-Track-5k ($336 \times 336$)dataset is available at https://huggingface.co/datasets/WangBiao/R1-Track-5k (For EasyR1) and https://huggingface.co/datasets/WangBiao/R1-Track-Data-ShareGPT (For Llamafactory). 
Note that this dataset was randomly sampled from [Got10k](http://got-10k.aitestunion.com/) and has not undergone manual review. Some image pairs are of relatively low quality, but we will address and improve this issue in the future.
<img width="500" alt="image" src="https://github.com/user-attachments/assets/25afecd3-16b9-4a02-a816-eb2a8bf63ba4" />


## Quick Start
A detailed user guide will be launched in the near future.

-SFT
```bash
Please refer to the official LLaMA-Factory repo for env configuration guidelines, and add the supplied datasets and scripts to the specified directories as outlined in the documentation.
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_lora/r1_track_lora_sft.yaml
```

```bash
llamafactory-cli export examples/merge_lora/r1_track_lora_sft.yaml
```

-RL
```bash
Please refer to the official EasyR1 repo for env configuration guidelines, and add the supplied datasets and scripts to the specified directories as outlined in the documentation.
```


## Some Findings
Our assembled fine-tuning dataset contains a critical flaw :scream: : all target objects in the images have nearly equal width-to-height ratios (1:1), making them effectively "square." This caused R1-Track-SFT to easily learn this appearance feature and overfit 
, leading to significant errors during video tracking. In contrast, R1-Track-GRPO avoided this issue :sunglasses:, likely through its reasoning process or soft supervision from GIoU rewards 
. While we plan to construct more balanced datasets in the future, this observation already demonstrates the advantages of reinforcement learning in mitigating dataset biases.


## Timeline
- [2025/04/02] **We released [R1-Track-5K](https://huggingface.co/datasets/WangBiao/R1-Track-5k) dataset!**
- [2025/04/20] **We released [R1-Track-SFT](https://huggingface.co/WangBiao/R1-Track-SFT) model!**
- [2025/04/24] **We released [R1-Track-Data-ShareGPT](https://huggingface.co/datasets/WangBiao/R1-Track-Data-ShareGPT) dataset. You can effortlessly integrate it with LlamaFactory for use!**
