import sys
sys.path.append('.')
import numpy as np
import traceback
import time
import csv
import json
import base64
import random
import asyncio
import aiohttp
import time
from time import sleep
import requests
import aiohttp
import cv2

from io import BytesIO
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()


# def convertToImage(img_binary) :
#     if img_binary is None:
#         return None
#     buffer = np.frombuffer(img_binary, np.uint8)
#     img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
#     return img

def convertToImage(img_binary) :
    if img_binary is None:
        return None
    buffer = np.frombuffer(img_binary, np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    
    if img is not None:
        return img
    else:
        try:
            image = Image.open(BytesIO(img_binary)).convert('RGB')  # rbg format pil image
            image = np.array(image)
            return image[..., ::-1]  # convert rgb to bgr
        except:
            print('convert img bins to pil image error')
            return None


async def download_img(url, with_preprocess, resize_size, encode):
    img_bin = None
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.get(url, ssl=False) as resp:
            if resp.status == 200:
                img_bin = await resp.content.read()
                img_bin = convertToImage(img_bin) if with_preprocess else img_bin
                if img_bin is not None:
                    img_bin = cv2.resize(img_bin, resize_size) if resize_size is not None else img_bin
                    img_bin = cv2.imencode('.png', img_bin)[1].tostring() if encode == True else img_bin
    return (url, img_bin)
 
    

async def async_img_downloader(imgurls, with_preprocess, resize_size, encode):
    futures = [asyncio.create_task(download_img(img_url, with_preprocess, resize_size, encode)) for img_url in imgurls]
    done, pending = await asyncio.wait(futures)

    allres = []
    errorquery = 0
    #for feature in futures:
    #    print(feature.result())
    for task in done:
        if task is not None and task.result() is not None:
            allres.append(task.result())
        else:
            errorquery += 1
    return allres, errorquery


