import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import argparse

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

def i2i_recommendation(target_item_id, note_embeds, note_ids, k=100):
    """
    Given a target item, find the most similar items
    
    Args:
        target_item_id: The target item ID
        note_embeds: Tensor of all item embeddings [num_items, embed_dim]  
        note_ids: Array of all item IDs
        k: Number of similar items to return
    
    Returns:
        List of k most similar item IDs (excluding the target item itself)
    """
    # Find the index of target item
    try:
        target_idx = np.where(note_ids == target_item_id)[0][0]
    except IndexError:
        print(f"Target item {target_item_id} not found in item embeddings")
        return []
    
    # Get target item embedding
    target_embed = note_embeds[target_idx:target_idx+1]  # [1, embed_dim]
    
    # Compute similarity with all items
    similarities = target_embed @ note_embeds.T  # [1, num_items]
    similarities = similarities.squeeze(0)  # [num_items]
    
    # Get top k+1 (including the target item itself)
    _, top_indices = similarities.topk(k+1, dim=-1)
    top_indices_cpu = top_indices.cpu().numpy()
    
    # Get similar item IDs and remove the target item itself
    similar_items = note_ids[top_indices_cpu].tolist()
    if target_item_id in similar_items:
        similar_items.remove(target_item_id)
    
    return similar_items[:k]

def batch_i2i_recommendation(target_item_ids, note_embeds, note_ids, k=100):
    """
    Batch process multiple target items for i2i recommendation
    
    Args:
        target_item_ids: List of target item IDs
        note_embeds: Tensor of all item embeddings [num_items, embed_dim]
        note_ids: Array of all item IDs
        k: Number of similar items to return for each target
    
    Returns:
        Dict mapping target_item_id -> list of similar items
    """
    results = {}
    
    # Find indices for all target items at once
    target_indices = []
    valid_target_ids = []
    
    for target_id in target_item_ids:
        try:
            target_idx = np.where(note_ids == target_id)[0][0]
            target_indices.append(target_idx)
            valid_target_ids.append(target_id)
        except IndexError:
            print(f"Warning: Target item {target_id} not found in embeddings, skipping...")
            results[target_id] = []
    
    if not target_indices:
        return results
    
    # Batch compute similarities
    target_embeds = note_embeds[target_indices]  # [batch_size, embed_dim]
    similarities = target_embeds @ note_embeds.T  # [batch_size, num_items]
    
    # Get top k+1 for each target
    _, top_indices = similarities.topk(k+1, dim=-1)  # [batch_size, k+1]
    top_indices_cpu = top_indices.cpu().numpy()
    
    # Process results for each target
    for i, target_id in enumerate(valid_target_ids):
        similar_item_ids = note_ids[top_indices_cpu[i]].tolist()
        
        # Remove the target item itself
        if target_id in similar_item_ids:
            similar_item_ids.remove(target_id)
        
        results[target_id] = similar_item_ids[:k]
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag_name', type=str, default='win10_query3_train_v3')
    args = parser.parse_args()
    
    # Load user target data
    user_target_path = 'eval/user_to_target_info.json'
    with open(user_target_path, 'r') as f:
        user_target = json.load(f)

    # Extract target notes for each user
    userid2target_note = {}
    all_target_items = set()
    
    i = 0
    for data in user_target:
        user_id = data["user_id"]
        target_notes = data['target_note_id']
        userid2target_note[user_id] = target_notes
        all_target_items.update(target_notes)
        i += 1
        if i > 1000:
            break
    
    print(f"Total users: {len(userid2target_note)}")
    print(f"Total unique target items: {len(all_target_items)}")
    
    # Load item embeddings (only once)
    model_name = args.tag_name
    item_embed_root = f'expr/{model_name}'
    
    print("Loading item embeddings...")
    with open(f"{item_embed_root}/base_item_embedding_pr.pkl", "rb") as f:
        note_embeds_data = pickle.load(f)
    print(f"Successfully loaded {len(note_embeds_data)} note embeddings")
    
    # Process note embeddings
    print("Processing embeddings...")
    note_embeds_list = []
    note_ids_list = []
    for note_data in note_embeds_data:
        note_embeds_list.append(note_data['embed'])
        note_ids_list.append(note_data['id'])
    
    # Convert to tensors and move to GPU
    note_embeds = torch.tensor(note_embeds_list, dtype=torch.float16).cuda()
    note_embeds = note_embeds / note_embeds.norm(dim=-1, keepdim=True)  # Normalize
    note_ids = np.array(note_ids_list)
    note_ids_set = set(note_ids_list)
    
    print(f"Processed {len(note_embeds)} note embeddings")
    print(f"Note embedding shape: {note_embeds.shape}")
    
    # Filter valid target items (items that exist in embeddings)
    valid_target_items = list(all_target_items & note_ids_set)
    print(f"Valid target items (in embeddings): {len(valid_target_items)}")
    
    # Batch process i2i recommendations
    print("Computing i2i recommendations...")
    i2i_results = batch_i2i_recommendation(valid_target_items, note_embeds, note_ids, k=100)
    
    # Evaluate results
    all_target_lists = []
    all_ranked_lists = []
    query_list = []
    
    for user_id, target_notes in tqdm(userid2target_note.items(), desc="Processing users"):
        user_targets = [item for item in target_notes if item in note_ids_set]
        if not user_targets:
            continue
            
        # For evaluation, we use the first target item to get recommendations
        # and check if other target items are in the recommendations
        if len(user_targets) < 2:
            continue  # Need at least 2 items for evaluation
            
        query_item = user_targets[0]
        ground_truth = user_targets[1:]  # Other items as ground truth
        
        if query_item in i2i_results:
            recommendations = i2i_results[query_item]
            query_list.append(query_item)
            all_target_lists.append(ground_truth)
            all_ranked_lists.append(recommendations)
    
    if all_target_lists:
        # Calculate metrics
        print("Calculating metrics...")
        metrics = batch_calculate_metrics(all_target_lists, all_ranked_lists)
        
        # Print results
        print('------- I2I Recommendation Results -------')
        print(f'Total evaluation cases: {len(all_target_lists)}')
        print(f'NDCG@10: {metrics["ndcg_10"].mean():.4f}')
        print(f'NDCG@100: {metrics["ndcg_100"].mean():.4f}')
        print(f'NDCG@1000: {metrics["ndcg_1000"].mean():.4f}')
        print(f'HR@10: {metrics["hr_10"].mean():.4f}')
        print(f'HR@100: {metrics["hr_100"].mean():.4f}')  
        print(f'HR@1000: {metrics["hr_1000"].mean():.4f}')
        print(f'MRR: {metrics["mrr"].mean():.4f}')
        
        # Show some example results
        print('\n------- Example Results -------')
        for i, (query, targets, recs) in enumerate(zip(query_list[:3], all_target_lists[:3], all_ranked_lists[:3])):
            print(f"Example {i+1}:")
            print(f"  Query: {query}")
            print(f"  Ground truth: {targets}")
            print(f"  Top 10 recommendations: {', '.join(recs[:10])}")
            print(f"  Hits in top 10: {[item for item in recs[:10] if item in targets]}")
            print()
    else:
        print("No valid evaluation cases found!")
    
    # Optionally save results
    # with open(f'i2i_results_{model_name}.json', 'w') as f:
    #     json.dump(i2i_results, f, indent=2)
