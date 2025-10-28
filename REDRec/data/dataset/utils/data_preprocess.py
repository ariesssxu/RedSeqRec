import os
import random
import numpy as np
import traceback
import json
import time
import requests

import requests
from io import BytesIO
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()


import torch
import torchvision.transforms as T

from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)




def note_content_preprocess(note_content, content_max_len=180, max_topictags=None):
    lists = note_content.split('#')
    content = ""
    topictags_lists = []
    contents = []
    for per in lists:
        per = per.strip()
        if per in ['', ',', '，', '.', '。']:
            continue
        if '[话题]' in per:
            topictags_lists.append(per)
        else:
            contents.append(per)
    
    topictags = []
    used_topictags = set()
    for tag in topictags_lists:
        if tag not in used_topictags:
            used_topictags.add(tag)
            topictags.append(tag)
    
    for idx, per in enumerate(contents):
        content += per
        if idx != len(contents) - 1:
            content += ','
    
    if max_topictags is not None:
        topictags = topictags[:max_topictags]

    if len(topictags):
        tag_content = "话题词："
        for idx, tag in enumerate(topictags):
            tag_content += tag.replace('[话题]', '')
            if idx != len(topictags) - 1:
                tag_content += ','
        tag_content += '。'
        content = tag_content + content

    content = content[:content_max_len]

    return content


def get_notelist_by_ts(note_list):
    ts = []
    note_ids = []
    for line in note_list:
        ts.append(line['ts'])
        note_ids.append(line['note_id'])
    
    ts = np.array(ts)
    note_ids = np.array(note_ids)
    # sort by ts
    sorted_index = ts.argsort()
    note_list = note_ids[sorted_index].tolist()
    return note_list


def get_lastn_info(note_list, lastn_note_appear_cnt):
    note_ids = []
    actions = []
    ts = []
    
    for per_note_info in note_list:
        note_id = per_note_info['note_id']
        
        if lastn_note_appear_cnt is not None:
            if note_id not in lastn_note_appear_cnt:
                print(f'{note_id} not in lastn_note_appear_cnt dict.')
                continue
            if drop_this_note(lastn_note_appear_cnt[note_id], min_val=285, max_val=2000):
                continue
        
        action_list = per_note_info['action']
        
        note_ids.append(note_id)
        actions.append(action_list)
        ts.append(per_note_info['ts'])

    return note_ids, actions, ts


def drop_this_note(appear_cnt, min_val=100, max_val=1000):
    if appear_cnt < min_val:
        return False
    
    drop_rate = appear_cnt / max_val
    
    if drop_rate > 1:
        return True

    rng = np.random.rand()
    if rng > 1 - drop_rate:
        return False
    return True


def random_drop_hot_note(note_list, note2cnt, thresh_val=50, max_val=300):
    new_note_list = []
    for note_id in note_list:
        if note_id not in note2cnt:
            new_note_list.append(note_id)
        else:
            cur_noteid_cnt = note2cnt[note_id]
            if cur_noteid_cnt >= thresh_val:
                drop_rate = cur_noteid_cnt / max_val
                rng = np.random.rand()
                if rng > 1 - drop_rate:
                    new_note_list.append(note_id)
            else:
                new_note_list.append(note_id)
    
    return new_note_list


    
def lastn_sort_by_ts(lastn):
    ts = []
    for per in lastn:
        ts.append(per['ts'])
    sorted_index = np.argsort(ts)
    return np.array(lastn)[sorted_index].tolist()


def convertToImage(img_binary):
    image = Image.open(BytesIO(img_binary)).convert('RGB')  # rbg format pil image
    return image


def download_img(url, resize_size, fix_wh_ratio=False):
    try:
        response = requests.get(url, timeout=5)
    except requests.exceptions.Timeout:
        print(f"Timeout: {url}")
        return None
    
    bins = response.content
    try:
        pil_img = convertToImage(bins)
    except:
        # print(f'{url} convert img bins to pil image error')
        return None
    
    if pil_img is not None:
        if not fix_wh_ratio:
            pil_img = pil_img.resize((resize_size, resize_size)) if resize_size is not None else pil_img
        else:
            max_side_len = resize_size
            w, h = pil_img.size
            wh_ratio = w / h
            if w > h:
                new_w = max_side_len
                new_h = new_w / wh_ratio
            else:
                new_h = max_side_len
                new_w = new_h * wh_ratio
            
            new_w, new_h = int(new_w), int(new_h)
            pil_img = pil_img.resize((new_w, new_h))

    return pil_img



def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform



def process_image(pil_img, input_size=448):
    image = pil_img
    transform = build_transform(input_size=input_size)
    pixel_values = transform(image)
    return pixel_values




def _convert_to_rgb(image):
    return image.convert('RGB')


def get_transform(img_size=224):
    transform = Compose([
                        Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
                        _convert_to_rgb,
                        ToTensor(),
                        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    return transform


if __name__ == '__main__':
    url = 'http://ci.xiaohongshu.com/1040g00831eea34j80s005nprqdf08hrg609b9p0'
    pil_img = download_img(url, 224, False)

    import pdb; pdb.set_trace()
