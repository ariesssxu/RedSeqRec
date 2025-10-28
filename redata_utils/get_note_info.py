import json
import logging
import os
from typing import List, Dict, Optional, Set
from tqdm import tqdm
import sys
import concurrent.futures
import threading
from functools import partial
from REDRec.data.dataset.utils.kv import NoteInfoCacher

class NoteContentSaver:
    def __init__(self, num_workers=10):
        self.logger = logging.getLogger(__name__)
        self.num_workers = num_workers
        self.thread_local = threading.local()
        
        # Define the train_noteinfo_kv configuration directly
        self.train_noteinfo_kv = {
            "ip": '',
            "port": '',
            "prefix": '',
            "ex": 1
        }
    
    def _get_noteinfo_cacher(self):
        if not hasattr(self.thread_local, 'noteinfo_cacher'):
            self.thread_local.noteinfo_cacher = NoteInfoCacher(**self.train_noteinfo_kv)
        return self.thread_local.noteinfo_cacher
    
    def extract_note_ids_from_json(self, json_path: str) -> Set[str]:
        """Extract all unique note IDs from the saved JSON file."""
        self.logger.info(f"Extracting note IDs from: {json_path}")
        
        homefeed_note_ids = set()
        ads_note_ids = set()
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each user's data
        for user_id, user_data in data.items():
            # Extract note IDs from homefeed
            if 'homefeed_noteid_list' in user_data:
                for item in user_data['homefeed_noteid_list']:
                    if 'note_id' in item:
                        homefeed_note_ids.add(item['note_id'])
            
            # Extract note IDs from ads
            if 'ads_click_noteid_list' in user_data:
                for item in user_data['ads_click_noteid_list']:
                    if 'note_id' in item:
                        ads_note_ids.add(item['note_id'])
        
        self.logger.info(f"Found {len(homefeed_note_ids)} unique homefeed note IDs")
        self.logger.info(f"Found {len(ads_note_ids)} unique ads note IDs")
        return homefeed_note_ids, ads_note_ids
    
    def get_note_content(self, note_id: str) -> Optional[Dict]:
        """Get note content for a single note ID."""
        try:
            noteinfo_cacher = self._get_noteinfo_cacher()
            note_info = noteinfo_cacher.get_note_info(note_id)
            note_info["note_id"] = note_id
            return note_info
        except Exception as e:
            self.logger.debug(f'{note_id} get noteinfo failed. Error: {str(e)}')
            return None
    
    def _process_batch(self, note_ids_batch: List[str]) -> List[Dict]:
        results = []
        for note_id in note_ids_batch:
            note_info = self.get_note_content(note_id)
            if note_info:
                results.append(note_info)
        return results
    
    def save_note_contents(self, note_ids: List[str], output_path: str, batch_size: int = 1000):
        """Save note contents to a file using multiple threads."""
        self.logger.info(f"Processing {len(note_ids)} note IDs with {self.num_workers} workers")
        
        # Convert set to list if needed
        if isinstance(note_ids, set):
            note_ids = list(note_ids)
        
        note_contents = []
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        batches = [note_ids[i:i+batch_size] for i in range(0, len(note_ids), batch_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_batch = {executor.submit(self._process_batch, batch): batch for batch in batches}
            
            with tqdm(total=len(batches), desc="Processing batches") as pbar:
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_results = future.result()
                    note_contents.extend(batch_results)
                    pbar.update(1)
        
        self.logger.info(f"Successfully retrieved content for {len(note_contents)} notes out of {len(note_ids)} total")
        
        # Save to file in JSON format
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(note_contents, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved note contents to {output_path}")
        return note_contents

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    input_json_path = ""
    output_dir = ""
    output_homefeed_path = f"{output_dir}/homefeeds_note_contents.json"
    output_ads_path = f"{output_dir}/ads_note_contents.json"
    
    os.makedirs(output_dir, exist_ok=True)
    
    saver = NoteContentSaver(num_workers=20)
    
    homefeed_note_ids, ads_note_ids = saver.extract_note_ids_from_json(input_json_path)
    
    print("Processing homefeed notes...")
    saver.save_note_contents(homefeed_note_ids, output_homefeed_path, batch_size=100)
    
    print("Processing ads notes...")
    saver.save_note_contents(ads_note_ids, output_ads_path, batch_size=100)

if __name__ == "__main__":
    main()
