import os
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


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


model_name = 'YOUR_MODEL_TAG'
note_embed_root = f'YOUR_NOTE_EMBED_ROOT'
user_embed_root = f'YOUR_USER_EMBED_ROOT'

user_lastn_path = 'YOUR_USER_LASTN_PATH'
taxo_path = 'YOUR_TAXO_PATH'

with open(taxo_path, 'r') as f: 
    noteid2taxonomy2 = json.load(f)


# read lastn
with open(user_lastn_path, 'r') as f:
    user_lastn = json.load(f)
    

userid2target_note = {}
for user_id in user_lastn:
    userid2target_note[user_id] = user_lastn[user_id][-1]['note_id']


# read note_embed
note_ids = []
note_embeds = []
for per in tqdm(read_data_from_parquet(note_embed_root, folder=True)):
    note_ids.append(per['note_id'])
    note_embeds.append(per['embed'])


# read user_embed
user_ids = []
user_embeds = []
for per in tqdm(read_data_from_parquet(user_embed_root, folder=True)):
    user_ids.append(per['user_id'])
    uese_embed = per['embed']
    user_embeds.append(uese_embed)



userid2embed = {}
for i in range(len(user_ids)):
    userid2embed[user_ids[i]] = user_embeds[i]


note_embeds = np.array(note_embeds)
note_ids = np.array(note_ids)
base_note_ids_set = set(note_ids)
note_embeds = torch.from_numpy(note_embeds)
note_embeds = note_embeds.cuda().half()



userid_nums = 0
top10_hit = 0
top50_hit = 0
top100_hit = 0

top100_taxo_nums = []

for idx, user_id in tqdm(enumerate(user_ids)):
    target_noteid = userid2target_note[user_id]

    if target_noteid not in base_note_ids_set:
        continue
    
    userid_nums += 1
    cur_user_user_embed = userid2embed[user_id]
    cur_user_user_embed = torch.from_numpy(cur_user_user_embed).cuda().half()
    cur_user_user_embed = cur_user_user_embed.reshape(-1, 64)

    
    curr_user_sims = cur_user_user_embed @ note_embeds.T
    '''
    curr_user_sims, _ = curr_user_sims.max(dim=-2)

    # # top10
    # top10_values, top10_index = curr_user_sims.topk(10, dim=-1)
    # top10_recalled_noteids = note_ids[top10_index.cpu().numpy()].tolist()
    # if target_noteid in top10_recalled_noteids:
    #     top10_hit += 1
    
    # # top50
    # top50_values, top50_index = curr_user_sims.topk(50, dim=-1)
    # top50_recalled_noteids = note_ids[top50_index.cpu().numpy()].tolist()
    # if target_noteid in top50_recalled_noteids:
    #     top50_hit += 1
    

    # top100
    top100_values, top100_index = curr_user_sims.topk(100, dim=-1)
    top100_recalled_noteids = note_ids[top100_index.cpu().numpy()].tolist()
    

    cur_user_recall_taxo = set()
    for per in top100_recalled_noteids:
        if per in noteid2taxonomy2:
            cur_user_recall_taxo.add(noteid2taxonomy2[per])

    top100_taxo_nums.append(len(cur_user_recall_taxo))
    '''
    cur_user_recall_taxo = set()
    for curr_user_sim in curr_user_sims:
        rc = 33
        top100_values, top100_index = curr_user_sim.topk(rc, dim=-1)
        top100_recalled_noteids = note_ids[top100_index.cpu().numpy()].tolist()
    
        for per in top100_recalled_noteids:
            if per in noteid2taxonomy2:
                cur_user_recall_taxo.add(noteid2taxonomy2[per])

    top100_taxo_nums.append(len(cur_user_recall_taxo))

    # if idx % 100 == 0:
    #     print(f'R100: {(top100_hit / userid_nums):.4f}, R1000: {(top1000_hit / userid_nums):.4f}, R10000: {(top10000_hit / userid_nums):.4f}')


print('----------------------------')
print(f'user embed root: {user_embed_root}')
print(f'total user nums: {userid_nums}')
print(f'note base nums: {len(note_embeds)}')

print(f'taxo_nums@top100: {(np.mean(top100_taxo_nums)):.4f}')