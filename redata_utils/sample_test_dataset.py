# output file example
# {
#   "user_id_1": {
#     "homefeed_noteid_list": [
#       {"ts": 1720073175, "note_id": "6665dd71000000000c01957d", "action_set": "00000000000000100000000010000010", "page_key": 0, "duration": 45},
#       ...
#     ],
#     "ads_click_noteid_list": [
#       {"note_id": "67fe055a000000001d0394ff"},
#       ...
#     ]
#   },
#   "user_id_2": {
#     ...
#   }
# }

# python sample_parquet_to_json.py \
#     --input /path/to/input.parquet \
#     --output /path/to/output.json \
#     --sample_size 1000 \
#     --min_homefeed 50 \
#     --min_ads 30 \
#     --seed 42

import os
import json
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import argparse
from tqdm import tqdm
import re
import random

def parse_complex_list(text):
    """Parse text format homefeed list into structured data"""
    if text == "[]" or text is None:
        return []
    
    pattern = r'\[(\d+),\s*([a-f0-9]+),\s*([01]+),\s*(\d+),\s*(\d+)\]'
    matches = re.findall(pattern, text)
    
    result = []
    for match in matches:
        timestamp = int(match[0])
        note_id = match[1]
        action_set = match[2]
        page_key = int(match[3])
        duration = int(match[4])
        
        item = {
            'ts': timestamp,
            'note_id': note_id,
            'action_set': action_set,
            'page_key': page_key,
            'duration': duration
        }
        result.append(item)
    
    return result

def parse_ad_list(text):
    """Parse text format ad click list into structured data"""
    if text == "[]" or text is None:
        return []
    
    pattern = r'([0-9a-f]{6,})'
    hex_strings = re.findall(pattern, text)
    
    result = []
    for hex_string in hex_strings:
        item = {'note_id': hex_string}
        result.append(item)
    
    return result

def read_parquet_batches(file_path, batch_size=1000):
    """Read parquet file in batches"""
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()

def process_row(row):
    """Process a single data row and return formatted data"""
    user_id = row.get('user_id', '')
    
    # Process homefeed list
    homefeed_list = row.get('homefeed_noteid_list')
    if homefeed_list is None:
        homefeed_list = []
    elif isinstance(homefeed_list, str):
        homefeed_list = parse_complex_list(homefeed_list)
    elif isinstance(homefeed_list, np.ndarray):
        # Already in array format, ensure consistent structure
        homefeed_list = homefeed_list.tolist()
    homefeed_list = homefeed_list[:512]
    homefeed_list = homefeed_list[::-1]
    
    # Process ad click list
    ads_list = row.get('ads_click_noteid_list')
    if ads_list is None:
        ads_list = []
    elif isinstance(ads_list, str):
        ads_list = parse_ad_list(ads_list)
    elif isinstance(ads_list, np.ndarray):
        # Already in array format, ensure consistent structure
        ads_list = ads_list.tolist()
    ads_list = ads_list[:512]
    ads_list = ads_list[::-1]
    
    return {
        'user_id': user_id,
        'homefeed_noteid_list': homefeed_list,
        'ads_click_noteid_list': ads_list
    }

def sample_data_from_parquet(input_path, output_path, sample_size=1000, 
                             min_homefeed_len=0, min_ads_len=0, seed=42):
    """
    Sample data from parquet file and generate JSON format dataset
    
    Parameters:
    - input_path: Path to input parquet file
    - output_path: Path to output JSON file
    - sample_size: Number of records to sample
    - min_homefeed_len: Minimum length requirement for homefeed list
    - min_ads_len: Minimum length requirement for ad click list
    - seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Starting to sample data from {input_path}...")
    
    # First read all records that meet the criteria
    all_eligible_records = []
    processed_count = 0
    
    for batch_df in tqdm(read_parquet_batches(input_path), desc="Reading data"):
        for _, row in batch_df.iterrows():
            processed_count += 1

            if len(all_eligible_records) > sample_size * 10:
                break
            
            try:
                processed_row = process_row(dict(row))
                
                # Filter based on list lengths
                homefeed_len = len(processed_row['homefeed_noteid_list'])
                ads_len = len(processed_row['ads_click_noteid_list'])
                
                if homefeed_len >= min_homefeed_len and ads_len >= min_ads_len:
                    all_eligible_records.append(processed_row)
            except Exception as e:
                print(f"Error processing row: {e}")
            
            if processed_count % 10000 == 0:
                print(f"Processed {processed_count} records, found {len(all_eligible_records)} eligible records")
    
    print(f"Found a total of {len(all_eligible_records)} eligible records")
    
    # If there are not enough eligible records, use them all
    final_sample_size = min(sample_size, len(all_eligible_records))
    
    # Random sampling
    if len(all_eligible_records) > sample_size:
        sampled_records = random.sample(all_eligible_records, final_sample_size)
    else:
        sampled_records = all_eligible_records
    
    print(f"Final sample size: {len(sampled_records)} records")
    
    # Convert to required JSON format
    output_dict = {}
    for record in sampled_records:
        user_id = record['user_id']
        output_dict[user_id] = {
            'homefeed_noteid_list': record['homefeed_noteid_list'],
            'ads_click_noteid_list': record['ads_click_noteid_list']
        }
    
    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"Sample data saved to {output_path}")
    
    # Return some statistics
    stats = {
        "total_processed": processed_count,
        "eligible_records": len(all_eligible_records),
        "sampled_records": len(sampled_records)
    }
    return stats

def main():
    parser = argparse.ArgumentParser(description='Sample data from Parquet file to generate JSON dataset')
    parser.add_argument('--input', type=str, default="*.parquet", help='Path to input Parquet file')
    parser.add_argument('--output', type=str, default="*.json", help='Path to output JSON file')
    parser.add_argument('--sample_size', type=int, default=10000, help='Number of samples to take')
    parser.add_argument('--min_homefeed', type=int, default=82, help='Minimum length for homefeed list')
    parser.add_argument('--min_ads', type=int, default=82, help='Minimum length for ad click list')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    stats = sample_data_from_parquet(
        args.input,
        args.output,
        args.sample_size,
        args.min_homefeed,
        args.min_ads,
        args.seed
    )
    
    print(stats)
    
if __name__ == "__main__":
    main()