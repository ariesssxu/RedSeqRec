import os

import torch

from REDRec.utils import get_model
from REDRec.data.dataset import REDRecEvalItemDataset, prepare_batchdata_for_note_inference
from zero_to_fp32 import load_state_dict_from_zero_checkpoint

import yaml
from tqdm import tqdm
from easydict import EasyDict as edict
from data_utils import save_local
from functools import partial
from transformers import AutoTokenizer

import argparse

def item_inputs_to_cuda(inputs, device):
    new_inputs = {}
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            new_inputs[key] = inputs[key].to(device)
        else:
            new_inputs[key] = inputs[key]
    return new_inputs


def worker(config_path, world_size, global_rank, device, tag_name):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)

    model_name = config.model.model_name
    model = get_model(model_name)(config)
    if config.eval.model_path.endswith('.bin'):
        state_dict = torch.load(config.eval.model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.bfloat16().eval()
    else:
        if os.path.isdir(config.eval.model_path):
            model = load_state_dict_from_zero_checkpoint(model, config.eval.model_path).bfloat16().eval()
        else:
            raise ValueError(f"Invalid model path: {config.eval.model_path}")
    model = model.to(device)

    # dataset config
    tokenizer = AutoTokenizer.from_pretrained(config.model.item_pretrain_dir, trust_remote_code=True)
    max_text_len = config.data.max_text_len
    max_topic_nums = config.data.max_topic_nums
    max_input_token_len = config.data.max_input_token_len
    item_prompt = config.data.item_prompt
    dataset = REDRecEvalItemDataset(global_rank, world_size, config, lastn_note=False)
    prepare_batchdata_for_note_inference = prepare_batchdata_for_note_inference

    dataloader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=64,
                                              shuffle=False,
                                              num_workers=8,
                                              collate_fn=partial(
                                                  prepare_batchdata_for_note_inference, 
                                                  tokenizer=tokenizer, 
                                                  max_text_len=max_text_len, 
                                                  max_topic_nums=max_topic_nums, 
                                                  max_input_token_len=max_input_token_len, 
                                                  item_prompt=item_prompt
                                                )
                                              )
    
    cnt = 0
    embeddings_to_save = []

    for sample in tqdm(dataloader):
        sample = item_inputs_to_cuda(sample, device)
        embed_2048d, embed_64d = model.compute_item(sample)
        
        embed_2048d = embed_2048d.cpu().float().numpy().tolist()
        embed_64d = embed_64d.cpu().float().numpy().tolist()
        note_ids = sample['note_ids']
        
        for idx in range(len(note_ids)):
            # raw_feature
            note_id = note_ids[idx]
            cur_raw_feature = embed_2048d[idx]
            cur_embed_64d = embed_64d[idx]
            
            cur_raw_feature = [round(per, 6) for per in cur_raw_feature]
            cur_embed_64d = [round(per, 6) for per in cur_embed_64d]

            
            embed_64d_msg = {
                            'id': note_id, 
                            'embed': cur_embed_64d,  
                            'embed_type': 'basepool_embed_64d'
                        }
            
            embeddings_to_save.append(embed_64d_msg)
            
            cnt += 1
            
            
    save_local(embeddings_to_save, f"eval/emb/{tag_name}/base_item_embedding_pr.pkl")

    print(f'worker_id: {global_rank} done, total note nums: {cnt}')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--global_rank', type=int, default=0)
    parser.add_argument('--config_path', type=str, default=0)
    parser.add_argument('--global_shift', type=int, default=0)
    parser.add_argument('--tag_name', type=str, default=0)
    args = parser.parse_args()

    config_path = args.config_path
    world_size = args.world_size
    global_rank = args.global_rank
    global_shift = args.global_shift
    tag_name = args.tag_name
    device = f'cuda:{global_rank + global_shift}'

    
    worker(
        config_path, world_size, global_rank, device, tag_name
    )