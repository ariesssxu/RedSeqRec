# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# Copyright (c) 2025 Xiaohongshu Technology Co. Ltd.
# SPDX-License-Identifier: MIT

# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

import os
import json
import argparse
import numpy as np
from easydict import EasyDict as edict
import yaml
from logging import getLogger

import torch
import torch.distributed as dist

from REDRec.data import bulid_dataloader
from REDRec.config import Config
from REDRec.utils import init_logger, get_model, init_seed, set_color
from REDRec.trainer import Trainer
from utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def convert_str(s):
    try:
        if s.lower() == 'none':
            return None
        if s.lower() == 'true':
            return True
        if s.lower() == 'false':
            return False
        float_val = float(s)
        if float_val.is_integer():
            return int(float_val)
        return float_val
    except ValueError:
        print(f"Unable to convert the string '{s}' to None / Bool / Float / Int, retaining the original string.")
        return s

def run_loop(local_rank, config_file, extra_args=[]):
    world_size = torch.distributed.get_world_size()
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)

    device = torch.device("cuda", local_rank)
    config['device'] = device

    # Parse extra --key value cli arguments
    for i in range(0, len(extra_args), 2):
        key = extra_args[i][2:]
        value = extra_args[i + 1]
        try:
            if '[' in value or '{' in value:
                value = json.loads(value)
                if isinstance(value, dict):
                    for k, v in value.items():
                        value[k] = convert_str(v)
                else:
                    value = [convert_str(x) for x in value]
            else:
                value = convert_str(value)
            if '.' in key:
                k1, k2 = key.split('.')
                config[k1][k2] = value
            else:
                config[key] = value
        except Exception as e:
            raise ValueError(f"{key} {value} invalid") from e

    # Seed, logger
    init_seed(config.get('seed', 2025), config.get('reproducibility', False))  
    init_logger(config)
    logger = getLogger()
    logger.info('Initialize root logger successfully!')

    logger.info('>>> config:')
    for key in config:
        logger.info(f'{key}: {config[key]}')

    # Model & data
    model_name = config.model.model_name
    model = get_model(model_name)(config)
    train_dl, valid_dl, test_dl = bulid_dataloader(config, local_rank, world_size)

    if config.training.get('load_pretrained_model', False):
        pretrained_path = config.training.load_pretrained_model
        logger.info(f'>>> load pretrained model from: {pretrained_path}')
        model = load_state_dict_from_zero_checkpoint(model, pretrained_path).bfloat16()

    trainer = Trainer(config, model)
    trainer.fit(train_dl, show_progress=config.get('show_progress', False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
        default='config/demo_multiscene.yaml')
    # Accept extra unknown args
    args, extra_args = parser.parse_known_args()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    config_file = args.config_path

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    run_loop(local_rank=local_rank, config_file=config_file, extra_args=extra_args)
