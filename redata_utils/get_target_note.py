import json
import argparse
import os
from typing import List, Dict, Any

def process_input_file(input_json_path: str, output_homefeed_json: str, output_ads_json: str) -> None:
    """
    Process input JSON file and generate output JSON file with last note_id and timestamp.
    
    Args:
        input_json_path: Path to input JSON file
        output_json_path: Path to output JSON file
    """
    # Read input JSON file
    print(f"Reading input file: {input_json_path}")
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    
    # Process homefeed data
    homefeed_results = []
    for user_id, user_data in data.items():
        homefeed_list = user_data.get('homefeed_noteid_list', [])
        if homefeed_list:
            # Get the last note in the sequence (should be most recent)
            last_note = homefeed_list[-1]  # Since the data is reversed in your sampling script
            note_id = last_note.get('note_id')
            timestamp = last_note.get('ts')
            
            if note_id and timestamp:
                homefeed_results.append({
                    'user_id': user_id,
                    'target_note_id': [note_id],
                    'target_timestamps': [timestamp]
                })
    
    # Process ads data
    ads_results = []
    for user_id, user_data in data.items():
        ads_list = user_data.get('ads_click_noteid_list', [])
        if ads_list:
            # Get the last ad in the sequence
            last_ad = ads_list[-1]  # Since the data is reversed in your sampling script
            
            # In your sample script, ads are just note_ids without timestamps
            if isinstance(last_ad, dict) and 'note_id' in last_ad:
                note_id = last_ad['note_id']
                # For ads, if we don't have timestamps, we'll use a placeholder
                # Use a fixed timestamp or omit it as needed
                timestamp = 1742411425  # Using example timestamp from your spec
            else:
                # If last_ad is directly the note_id string
                note_id = last_ad
                timestamp = 1742411425  # Using example timestamp from your spec
            
            ads_results.append({
                'user_id': user_id,
                'target_note_id': [note_id],
                'target_timestamps': [timestamp]
            })
    
    # Combine results
    all_results = homefeed_results + ads_results
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_homefeed_json)), exist_ok=True)
    
    # Write output JSON file
    print(f"Writing output file: {output_homefeed_json}")
    with open(output_homefeed_json, 'w') as f:
        json.dump(homefeed_results, f, ensure_ascii=False)
    
    print(f"Processing complete. Output saved to {output_homefeed_json}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_ads_json)), exist_ok=True)
    
    # Write output JSON file
    print(f"Writing output file: {output_ads_json}")
    with open(output_ads_json, 'w') as f:
        json.dump(ads_results, f, ensure_ascii=False)
    
    print(f"Processing complete. Output saved to {output_ads_json}")

    print(f"Total records: {len(all_results)} (Homefeed: {len(homefeed_results)}, Ads: {len(ads_results)})")

def main():
    parser = argparse.ArgumentParser(description='Process JSON data to extract last note_id and timestamp')
    parser.add_argument('--input_json', type=str, default="eval.json", help='Path to input JSON file')
    parser.add_argument('--output_homefeed_json', type=str, default="eval_target_noteid_info_homefeed.json", help='Path to output JSON file')
    parser.add_argument('--output_ads_json', type=str, default="eval_target_noteid_info_ads.json", help='Path to output JSON file')

    args = parser.parse_args()
    
    process_input_file(args.input_json, args.output_homefeed_json, args.output_ads_json)

if __name__ == "__main__":
    main()
