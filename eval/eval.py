import os
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import argparse

def str_to_list(s):
    # Remove possible brackets and whitespace
    s = s.strip().strip('[]')
    # Split by comma and strip spaces for each item
    return [item.strip() for item in s.split(',') if item.strip()]

def calculate_ndcg_at_k(target_items, recommended_items, k=10):
    """NDCG@k"""
    if not target_items:
        return 0.0
    
    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        if item in target_items:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0
    
    idcg = 0.0
    for i in range(min(k, len(target_items))):
        idcg += 1.0 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

def calculate_mrr(target_items, recommended_items):
    """MRR (Mean Reciprocal Rank)"""
    for i, item in enumerate(recommended_items, 1):
        if item in target_items:
            return 1.0 / i
    return 0.0

def filter_imp_data(user_id, input_pool, df, click=False):
    if isinstance(input_pool, str):
        pool_list = [item.strip() for item in input_pool.strip().strip('[]').split(',') if item.strip()]
    else:
        pool_list = list(map(str, input_pool))
    
    user_row = df[df['user_id'] == user_id]
    if user_row.empty:
        return 0, []
    
    imp_note_str = list(user_row['imp_note_ids'])[0] if not click else list(user_row['imp_click_note_ids'])[0]
    user_notes = str_to_list(imp_note_str)
    
    intersection = list(set(user_notes) & set(pool_list))
    return intersection


def merge_similar_queries(queries, threshold=0.98):
    assert queries.shape[0] == 3, "Input should contain exactly 3 queries"

    #norm_queries = F.normalize(queries, dim=1)
    sim_matrix = torch.matmul(queries, queries.T)

    merged = [False] * 3
    groups = []

    for i in range(3):
        if merged[i]:
            continue
        group = [i]
        for j in range(i + 1, 3):
            if not merged[j] and sim_matrix[i, j] > threshold:
                group.append(j)
                merged[j] = True
        merged[i] = True
        groups.append(group)

    merged_queries = torch.stack([queries[group].mean(dim=0) for group in groups])
    return merged_queries, len(merged_queries)

def batch_calculate_metrics(target_lists, ranked_lists, k_values=[10, 50, 100, 1000]):
    batch_size = len(target_lists)
    
    metrics = {
        'hr_10': np.zeros(batch_size),
        'hr_100': np.zeros(batch_size),  
        'hr_1000': np.zeros(batch_size),
        'ndcg_10': np.zeros(batch_size),
        'ndcg_100': np.zeros(batch_size),
        'ndcg_1000': np.zeros(batch_size),
        'mrr': np.zeros(batch_size)
    }
    
    for i in range(batch_size):
        targets = set(target_lists[i])
        if not targets:
            continue
            
        ranked = ranked_lists[i]
        
        for k in [10, 100, 1000]:
            if any(item in targets for item in ranked[:k]):
                metrics[f'hr_{k}'][i] = 1
        
        for k in [10, 100, 1000]:
            dcg = sum(1.0 / np.log2(pos + 2) for pos, item in enumerate(ranked[:k]) if item in targets)
            idcg = sum(1.0 / np.log2(pos + 2) for pos in range(min(k, len(targets))))
            metrics[f'ndcg_{k}'][i] = dcg / idcg if idcg > 0 else 0.0
        
        for pos, item in enumerate(ranked, 1):
            if item in targets:
                metrics['mrr'][i] = 1.0 / pos
                break
    
    return metrics

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tag_name', type=str, default=0)

    args = parser.parse_args()

    model_name = args.tag_name
    note_embed_root = f'emb/{model_name}'
    user_embed_root = f'emb/{model_name}'

    user_target_path = 'user_lastn.json'
    taxo2_test = False
    
    if taxo2_test:
        taxo2_path = 'noteid2taxonomy2.json'
        with open(taxo2_path, 'r') as f:
            noteid2taxo2 = json.load(f)
        
    # read lastn
    with open(user_target_path, 'r') as f:
        user_target = json.load(f)

    userid2target_note = {}
    for user_id, data in user_target.items():
        target_notes = [n["note_id"] for n in data['ads_click_noteid_target']]
        userid2target_note[user_id] = target_notes

    # read note_embed
    note_ids = []
    note_embeds = []
    # for per in tqdm(read_data_from_parquet(note_embed_root, folder=True)):
    #     note_ids.append(per['note_id'])
    #     note_embeds.append(per['embed'])
    with open(f"{note_embed_root}/base_item_embedding_pr.pkl", "rb") as f:
        note_embeds = pickle.load(f)
    print("Successfully load note_embeds, Len:", len(note_embeds))

    # read user_embed
    user_ids = []
    user_embeds = []
    # for per in tqdm(read_data_from_parquet(user_embed_root, folder=True)):
    #     user_ids.append(per['user_id'])
    #     uese_embed = per['embed']
    #     user_embeds.append(uese_embed)
    with open(f"{user_embed_root}/user_embedding_pr.pkl", "rb") as f:
        user_embeds = pickle.load(f)
    print("Successfully load user_embeds, Len:", len(user_embeds))

    # Read user embeddings
    userid2embed = {}
    for user_data in user_embeds:
        user_id = user_data['id']
        user_ids.append(user_id)
        # Extract embedding and convert to tensor
        embed_tensor = torch.tensor(user_data['embed'], dtype=torch.float16).cuda()
        # Normalize embedding
        embed_tensor = embed_tensor / embed_tensor.norm(dim=-1, keepdim=True)
        userid2embed[user_id] = embed_tensor

    # Process note embeddings
    note_embeds_list = []
    note_ids_list = []
    for note_data in note_embeds:
        note_embeds_list.append(note_data['embed'])
        note_ids_list.append(note_data['id'])

    # Convert to arrays and move to GPU
    note_embeds = torch.tensor(note_embeds_list, dtype=torch.float16).cuda()
    note_embeds = note_embeds / note_embeds.norm(dim=-1, keepdim=True)
    note_ids = np.array(note_ids_list)
    base_note_ids_set = set(note_ids)

    print(f"Processed {len(userid2embed)} user embeddings")
    print(f"Processed {len(note_embeds)} note embeddings")
    print(f"Note embedding shape: {note_embeds.shape}")

    batch_size = 256 
    all_user_ids = list(userid2embed.keys())
    
    valid_users = []
    valid_targets = []
    
    for user_id in all_user_ids:
        targets = userid2target_note.get(user_id, [])
        valid_user_targets = [t for t in targets if t in base_note_ids_set]
        if valid_user_targets:
            valid_users.append(user_id)
            valid_targets.append(valid_user_targets)
    
    print(f"Valid users for evaluation: {len(valid_users)}")
    
    all_metrics = []
    
    for batch_start in tqdm(range(0, len(valid_users), batch_size), desc="Batch processing"):
        batch_end = min(batch_start + batch_size, len(valid_users))
        batch_users = valid_users[batch_start:batch_end]
        batch_targets = valid_targets[batch_start:batch_end]
        
        batch_similarities = []
        
        for user_id in batch_users:
            cur_user_user_embed = userid2embed[user_id]
            cur_user_user_embed = cur_user_user_embed.cuda().half()
            cur_user_user_embed = cur_user_user_embed.view(-1, 64)  # [num_user_embeds, 64]
            
            curr_user_sims = cur_user_user_embed @ note_embeds.T  # [num_user_embeds, num_notes]
            curr_user_sims = torch.max(curr_user_sims, dim=-2)[0]  # [num_notes]
            
            batch_similarities.append(curr_user_sims)
        
        batch_similarities = torch.stack(batch_similarities, dim=0)  # [batch_size, num_notes]
        
        max_k = 1000
        _, batch_top_indices = batch_similarities.topk(max_k, dim=-1)
        
        batch_top_indices_cpu = batch_top_indices.cpu().numpy()
        batch_ranked_lists = []
        
        for i in range(len(batch_users)):
            ranked_note_ids = note_ids[batch_top_indices_cpu[i]].tolist()
            batch_ranked_lists.append(ranked_note_ids)
        
        batch_metrics = batch_calculate_metrics(batch_targets, batch_ranked_lists)
        all_metrics.append(batch_metrics)
    
    final_metrics = {}
    for key in ['hr_10', 'hr_100', 'hr_1000', 'ndcg_10', 'ndcg_100', 'ndcg_1000', 'mrr']:
        final_metrics[key] = np.concatenate([m[key] for m in all_metrics])
    
    print('------- Compatible Optimized Results -------')
    print(f'Total valid users: {len(valid_users)}')
    print(f'NDCG@10: {final_metrics["ndcg_10"].mean():.4f}')
    print(f'NDCG@100: {final_metrics["ndcg_100"].mean():.4f}')
    print(f'NDCG@1000: {final_metrics["ndcg_1000"].mean():.4f}')
    print(f'HR@10: {final_metrics["hr_10"].mean():.4f}')
    print(f'HR@100: {final_metrics["hr_100"].mean():.4f}')  
    print(f'HR@1000: {final_metrics["hr_1000"].mean():.4f}')
    print(f'MRR: {final_metrics["mrr"].mean():.4f}')