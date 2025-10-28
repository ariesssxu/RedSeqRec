#/usr/bin/env python
# -*- coding: UTF-8 -*-
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import csv
import os
import time
import random
import collections
import sys
sys.path.append('.')
import torch
import traceback
from pandas import read_parquet
from pathlib import Path
import json
import requests
import numpy as np
import cv2
import base64

import time
import numpy as np
import threading
import aiohttp
import asyncio
import multiprocessing
from lib.async_downloader import async_img_downloader
#from async_downloader import async_img_downloader

try:
    import queue
except ImportError:
    import Queue as queue


def read_data_from_parquet(filepath, header=None, select_col=None, folder=True):
    if folder:
        for root, dirs, files in os.walk(filepath):
            for file in files:
                if 'parquet' not in file:
                    continue
                file_path = os.path.join(root, file)
                data = read_parquet(file_path)
                if header is None:
                    header = data.columns.tolist()
                    select_col = header

                for indexs in data.index:
                    row = data.loc[indexs].values
                    info = {}
                    for col in select_col:
                        index = header.index(col)
                        info[col] = row[index]
                    yield info
    else:
        data = read_parquet(filepath)
        if header is None:
            header = data.columns.tolist()
            select_col = header

        for indexs in data.index:
            row = data.loc[indexs].values
            info = {}
            for col in select_col:
                index = header.index(col)
                info[col] = row[index]
            yield info 


def download(url) :
    try:
        for i in range(3):
            resp = requests.get(url)
            if resp.ok : 
                return resp.content
        return None
    except:
        return None

def convertToImage(img_binary) : 
    if img_binary is None:
        return None
    buffer = np.frombuffer(img_binary, np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return img

def convertBase64ToImage(imgstring, size=224):
    t1 = time.time()
    image = np.frombuffer(imgstring, dtype=np.uint8)
    image = image.reshape(size, size, 3) 
    t2 = time.time()
    print(t2 -t1)
    return image


def get_ocr_info(info):
    try:
        ocr_info = json.loads(info)
        ocr_texts = {}
        if "resourceResult" in ocr_info and "modelResult" in ocr_info['resourceResult'] and "detectResponse" in ocr_info['resourceResult']['modelResult'] and "detectInfos" in ocr_info['resourceResult']['modelResult']["detectResponse"]:
            for detectInfo in ocr_info['resourceResult']['modelResult']["detectResponse"]["detectInfos"]:
                try:
                    if "labelInfos" in detectInfo:
                        if detectInfo["labelInfos"][0]["info"] not in ocr_texts:
                            ocr_texts[detectInfo["labelInfos"][0]["info"]] = detectInfo["areaInfo"]["areaRatio"]
                        else:
                            if ocr_texts[detectInfo["labelInfos"][0]["info"]] < detectInfo["areaInfo"]["areaRatio"]:
                                ocr_texts[detectInfo["labelInfos"][0]["info"]] = detectInfo["areaInfo"]["areaRatio"]
                except:
                    print(traceback.format_exc())
                    continue
        sorted_ocr_info = sorted(ocr_texts.items(), key=lambda x: x[1], reverse=True)

        sorted_ocr_texts = ""
        for ocr_info in sorted_ocr_info:
            sorted_ocr_texts += ocr_info[0]
        return sorted_ocr_texts
    except:
        return ''

def download_single_image(img_url):
    return img_url, convertToImage(download(img_url))

def download_images_gevent(file_url, with_preprocess=True, resize_size=(224, 224), encode=True):
    try:
        multi_results = []
        for url in file_url:
            event = gevent.spawn(download_single_image, url)
            multi_results.append(event)
        res = gevent.joinall(multi_results)
        imgs = {}
        for i in res:
            tmp_res = i.value
            if tmp_res:
                url, img_bin = tmp_res
                if img_bin is not None:
                    img =  cv2.resize(img_bin, resize_size) if resize_size is not None else img_bin
                    imgs[url] =  cv2.imencode('.png', img)[1].tostring() if encode == True else img
        return None if len(imgs) == 0 else imgs
    except:
        print(traceback.format_exc(), file_url)
        return None

def convertToImage(img_binary) :
    if img_binary is None:
        return None
    buffer = np.frombuffer(img_binary, np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return img

def download_single_img(url, with_preprocess, resize_size, encode=True):
    img_bin = download(url) 
    if img_bin is None:
        return None
    img_bin = convertToImage(img_bin) if with_preprocess else img_bin
    if img_bin is not None:
        img_bin = cv2.resize(img_bin, resize_size) if resize_size is not None else img_bin
        img_bin = cv2.imencode('.png', img_bin)[1].tostring() if encode == True else img_bin
    return img_bin

def download_images(file_url, with_preprocess=True, resize_size=(224, 224), encode=True):
    try:
        img_bins, error = asyncio.run(async_img_downloader(file_url, with_preprocess, resize_size, encode))
        imgs = {img_url: img_bin for (img_url, img_bin) in img_bins if img_bin is not None}
        return None if len(imgs) == 0 else imgs
    except:
        print(traceback.format_exc(), file_url)
        return None

