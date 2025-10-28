import os
import pickle
import torch
from REDRec.utils import get_model
from REDRec.data.dataset import REDRecEvalUserDataset, user_dataset_collator
from zero_to_fp32 import load_state_dict_from_zero_checkpoint
from data_utils import save_local
import yaml
from tqdm import tqdm
from easydict import EasyDict as edict
import pandas as pd
import numpy as np
import argparse


def read_data_from_parquet(root, folder=True):
    if folder:
        for file in os.listdir(root):
            if 'parquet' not in file:
                continue
            
            file_path = os.path.join(root, file)
            data = pd.read_parquet(file_path)
            columns = list(data.columns)
            header = columns
            res = []
            for indexs in data.index:
                row = data.loc[indexs].values
                info = {}
                for col in columns:
                    index = header.index(col)
                    info[col] = row[index]
                
                yield info
    else:
        file_path = root
        data = pd.read_parquet(file_path)
        columns = list(data.columns)
        header = columns
        res = []
        for indexs in data.index:
            row = data.loc[indexs].values
            info = {}
            for col in columns:
                index = header.index(col)
                info[col] = row[index]
            
            yield info



def worker(config_path, device, tag_name):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)

    print('loading features...')

    note_embed_root = f'{tag_name}'
    with open(f"emb/{note_embed_root}/lastn_item_embedding_pr.pkl", "rb") as f:
        note_embeds = pickle.load(f)
    print("Successfully load note_embeds, Len:", len(note_embeds))
    noteids = []
    raw_embeds = []
    for note_data in note_embeds:
        raw_embeds.append(np.array(note_data['embed']))
        noteids.append(note_data['id'])

    noteids = np.array(noteids)

    note_id2idx = {}
    for i in range(len(noteids)):
        note_id2idx[noteids[i]] = i

    #import pdb; pdb.set_trace()
    

    model_name = config.model.model_name
    model = get_model(model_name)(config)
    model = load_state_dict_from_zero_checkpoint(model, config.eval.model_path).bfloat16().eval()

    model = model.to(device)

    dataset = REDRecEvalUserDataset(config)

    dataloader = torch.utils.data.DataLoader(dataset, 
                                            #   batch_size=config.eval.user_eval.user_eval_batch_size,
                                              batch_size=64,
                                              shuffle=False,
                                              num_workers=8,
                                              collate_fn=user_dataset_collator
                                              )
    
    cnt = 0
    embeddings_to_save = []
    
    for sample in tqdm(dataloader):
        user_ids = sample['user_ids']
        # compute_user_embedding(self, batch, note_id2idx, item_emb_tokens, item_llm, device):
        ret = model.compute_user_embedding(sample, note_id2idx, raw_embeds, device)

        user_embed_final_64d_norm = ret['user_embed_final_64d_norm']
        # action_pred_final_logits = ret['action_pred_final_logits']

        for idx in range(len(user_ids)):
            user_id = user_ids[idx]
            cur_user_user_embed = user_embed_final_64d_norm[idx]
            
            msg = {
                    'id': user_id, 
                    'embed': cur_user_user_embed,  
                    'embed_type': 'user_embed'
                }

            embeddings_to_save.append(msg)
            
            cnt += 1
    
    save_local(embeddings_to_save, f"emb/{tag_name}/user_embedding_pr.pkl")

    print(f'done, total user nums: {cnt}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--global_rank', type=int, default=0)
    parser.add_argument('--global_shift', type=int, default=0)
    parser.add_argument('--config_path', type=str, default=0)
    parser.add_argument('--tag_name', type=str, default=0)

    args = parser.parse_args()

    tag_name = args.tag_name
    config_path = args.config_path

    device = 'cuda:0'
    worker(config_path, device, tag_name)