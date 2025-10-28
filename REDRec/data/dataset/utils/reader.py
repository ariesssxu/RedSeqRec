import os
import json
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