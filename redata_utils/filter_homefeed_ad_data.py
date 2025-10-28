import os
import json
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import pyarrow.parquet as pq
import argparse
import ast
import re

def parse_complex_list(text):
    if text == "[]":
        return []
    pattern = r'\[(\d+),\s*([a-f0-9]+),\s*([01]+),\s*(\d+),\s*(\d+)\]'
    
    matches = re.findall(pattern, text)
    
    result = []
    for match in matches:
        timestamp = int(match[0])
        hex_value = match[1]
        binary_value = match[2]
        zero = int(match[3])
        count = int(match[4])
        
        item = {
            'ts': timestamp,
            'note_id': hex_value,
            'action_set': binary_value,
            'page_key': zero,
            'duration': count
        }
        result.append(item)
    
    return result

def parse_ad_list(text):
    """
    """
    pattern = r'([0-9a-f]{6,})'
    
    hex_strings = re.findall(pattern, text)
    
    result = []
    for hex_string in hex_strings:
        item = {'note_id': hex_string}
        result.append(item)
    
    return result

def read_data_from_parquet(file_path, batch_size=1000):
    """
    Generator that yields rows from a Parquet file as dictionaries.
    """
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        columns = list(df.columns)
        for row in df.itertuples(index=False, name=None):
            yield dict(zip(columns, row))

def worker(worker_id, save_root, data_root, file_list):
    """
    Worker process to filter and save users with:
    - homefeed_noteid_list length > 50
    - ads_click_noteid_list length > 30
    """
    filtered_data = []
    total_users = 0
    filtered_users = 0

    for filename in file_list:
        file_path = os.path.join(data_root, filename)
        for row in tqdm(read_data_from_parquet(file_path), desc=f"Worker {worker_id} processing {filename}"):
            total_users += 1
            
            homefeed_list = row.get('homefeed_noteid_list')
            if homefeed_list is None:
                homefeed_list = []
            elif isinstance(homefeed_list, str):
                homefeed_list = parse_complex_list(homefeed_list)
            
            ads_click_list = row.get('ads_click_noteid_list')
            if ads_click_list is None:
                ads_click_list = []
            elif isinstance(ads_click_list, str):
                ads_click_list = parse_ad_list(ads_click_list)
            
            if len(homefeed_list) > 82 and len(ads_click_list) > 82:
                filtered_users += 1
                
                row['homefeed_noteid_list'] = np.array(homefeed_list[:512])
                row['ads_click_noteid_list'] = np.array(ads_click_list)
                
                filtered_data.append(row)
            
            if total_users % 10000 == 0:
                print(f"Worker {worker_id}: Processed {total_users} users, filtered {filtered_users} users.")
                
                if len(filtered_data) >= 5000:
                    save_to_parquet(filtered_data, save_root, worker_id, batch_num=total_users//10000)
                    filtered_data = []
    
    if filtered_data:
        save_to_parquet(filtered_data, save_root, worker_id, batch_num="final")
    
    result = {
        'total_users': total_users,
        'filtered_users': filtered_users
    }
    
    stats_path = os.path.join(save_root, f'worker_{worker_id}_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"Worker {worker_id} done. Total users: {total_users}, Filtered users: {filtered_users}")

def save_to_parquet(data_list, save_root, worker_id, batch_num):
    """
    """
    if not data_list:
        return
        
    df = pd.DataFrame(data_list)
    save_path = os.path.join(save_root, f'filtered_data_worker_{worker_id}_batch_{batch_num}.parquet')
    df.to_parquet(save_path, index=False)
    print(f"Saved {len(data_list)} filtered records to {save_path}")

def main(worker_nums, save_root, data_root, split_file_list):
    processes = []
    for i in range(worker_nums):
        p = mp.Process(
            target=worker,
            args=(i, save_root, data_root, split_file_list[i].tolist())
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # 合并所有worker的统计信息
    total_processed = 0
    total_filtered = 0

    for i in range(worker_nums):
        stats_file = os.path.join(save_root, f'worker_{i}_stats.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                data = json.load(f)
                total_processed += data.get('total_users', 0)
                total_filtered += data.get('filtered_users', 0)
        else:
            print(f"Worker {i}'s stats file not found.")

    total_result = {
        'total_processed_users': total_processed,
        'total_filtered_users': total_filtered,
        'filter_conditions': {
            'min_homefeed_length': 50,
            'min_ads_click_length': 30
        }
    }

    total_path = os.path.join(save_root, 'total_stats.json')
    with open(total_path, 'w') as f:
        json.dump(total_result, f, indent=4)

    print(f"\nTotal users processed: {total_processed}")
    print(f"Users meeting filter conditions: {total_filtered}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter users with homefeed length >50 and ads click length >30')
    parser.add_argument('--worker_nums', type=int, default=1, help='Number of multiprocessing workers')
    args = parser.parse_args()

    data_root = ''
    save_root = ''

    os.makedirs(save_root, exist_ok=True)

    parquet_files = [f for f in os.listdir(data_root) if f.endswith('.parquet')]
    file_array = np.array(parquet_files)
    split_files = np.array_split(file_array, args.worker_nums)

    main(args.worker_nums, save_root, data_root, split_files)
