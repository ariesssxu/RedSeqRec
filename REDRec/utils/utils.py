# Copyright (c) 2024 westlake-repl
# SPDX-License-Identifier: MIT

import datetime
import importlib
import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    torch.distributed.barrier()
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur


def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_model(model_name):
    model_module = None

    if model_name == 'REDRec':
        module_path = '.'.join(['REDRec','model','redrec'])
        model_name = 'REDRec'
        
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)
    
    if model_module is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))

    model_class = getattr(model_module, model_name)
    return model_class


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value >= best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value <= best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def calculate_valid_score(valid_result, valid_metric=None):
    r""" return valid score from valid result

    Args:
        valid_result (dict): valid result
        valid_metric (str, optional): the selected metric in valid result for valid score

    Returns:
        float: valid score
    """
    if valid_metric:
        return valid_result[valid_metric]
    else:
        return valid_result['Recall@10']


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    return '    '.join([str(metric) + ' : ' + str(value) for metric, value in result_dict.items()])


def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def create_tensorboard(base_root):
    base_root = os.path.join(base_root, 'tensorboard')
    if not os.path.exists(base_root):
        os.makedirs(base_root)
    dir_name = '{}-{}'.format('model', get_local_time())
    dir_path = os.path.join(base_root, dir_name)
    writer = SummaryWriter(dir_path)
    return writer


def get_gpu_usage(device=None):
    r""" Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    return '{:.2f} G/{:.2f} G'.format(reserved, total)
