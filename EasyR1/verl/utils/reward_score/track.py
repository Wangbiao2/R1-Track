import re
from typing import Dict
import json

def check_format(s):
    pattern = r'^<think>(?:(?!<think>|</think>).)*</think>(?:(?!<think>|</think>|<answer>|</answer>).)*<answer>(?:(?!<answer>|</answer>).)*</answer>$'
    if re.fullmatch(pattern, s, flags=re.DOTALL):
        return 1.0
    else:
        return 0.0

def check_and_extract(s):
    pattern = r'^<think>.*</think>.*<answer>(.*?)</answer>$'
    match = re.fullmatch(pattern, s, flags=re.DOTALL)
    if match:
        return match.group(1)
    else:
        return 0.0


def calculate_giou(box1, box2):
    inter_xmin = max(box1[0], box2[0])
    inter_ymin = max(box1[1], box2[1])
    inter_xmax = min(box1[2], box2[2])
    inter_ymax = min(box1[3], box2[3])
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area != 0 else 0.0
    
    c_xmin = min(box1[0], box2[0])
    c_ymin = min(box1[1], box2[1])
    c_xmax = max(box1[2], box2[2])
    c_ymax = max(box1[3], box2[3])
    
    c_area = (c_xmax - c_xmin) * (c_ymax - c_ymin)
    
    giou = iou - (c_area - union_area) / c_area
    
    return giou


def track_compute_score(predict_str: str, ground_truth: str, response_length) -> Dict[str, float]:
    if not predict_str or not ground_truth:
        return {"overall": -1.0, "format": 0.0, "giou": 0.0, "length": 0.0}

    format_is_ok = check_format(predict_str)
    if not format_is_ok:
        return {"overall": -1.0, "format": 0.0, "giou": 0.0, "length": 0.0}
    else:
        format_reward = 1.0

    if response_length < 96:
        return {"overall": -1.0, "format": 0.0, "giou": 0.0, "length": 0.0}

    predict_str = check_and_extract(predict_str)
    try: 
        pre_bbox = json.loads(predict_str)
        gt_bbox = json.loads(ground_truth)
    except:
        return {"overall": -1.0, "format": 0.0, "giou": 0.0, "length": 0.0}
    
    try:
        giou_reward = calculate_giou(pre_bbox, gt_bbox)
    except:
        giou_reward = -1.0

    giou_reward_copy = giou_reward
    if giou_reward > 0 and giou_reward < 0.4:
        giou_reward = 0.0
    elif giou_reward > 0.75 and giou_reward < 0.95:
        giou_reward += 0.2
    elif giou_reward > 0.95:
        giou_reward += 0.5
    
    if response_length > 384:
        length_reward = (384 - response_length) / (512 - 384)
    else:
        length_reward = 0.0

        
    return {
        "overall": giou_reward + 0.1*format_reward + 0.5*length_reward,
        "format": format_reward,
        "giou": giou_reward_copy,
        "length": length_reward
    }
