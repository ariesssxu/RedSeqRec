import os
import json
from tqdm import tqdm

data_root = 'lastn_notes_appear_cnt/'
file_list = os.listdir(data_root)

note_id2cnt = {}
for file in tqdm(file_list):
    file_path = os.path.join(data_root, file)
    with open(file_path, 'r') as f:
        file_lines = json.load(f)
    
    for note_id in file_lines:
        if note_id not in note_id2cnt:
            note_id2cnt[note_id] = 0
        note_id2cnt[note_id] += file_lines[note_id]


# 统计所有note_id出现的比例
sorted_noteid2cnt = sorted(note_id2cnt.items(), key=lambda x:x[1])
sorted_noteid2cnt = sorted_noteid2cnt[::-1]
sorted_noteid2cnt = dict(sorted_noteid2cnt)
save_path = 'lastn_notes_appear_cnt/noteid2cnt.json'
with open(save_path, 'w') as f: 
    json.dump(sorted_noteid2cnt, f, indent=4)

import pdb; pdb.set_trace()

all_cnts = list(sorted_noteid2cnt.values())

# all_cnts = np.array(all_cnts)
# np.percentile(all_cnts, [50, 80, 90, 95, 96, 97, 98, 99])
# array([  2.,   8.,  21.,  53.,  70.,  98., 152., 306.])

