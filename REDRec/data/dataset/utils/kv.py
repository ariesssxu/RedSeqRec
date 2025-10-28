import os
import traceback
import json
import time
import lmdb
import pickle
from io import BytesIO
from PIL import Image

class LMDBClient(object):
    def __init__(self, db_path, map_size=1024*1024*1024*10):  # 10GB default
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.env = lmdb.open(db_path, map_size=map_size)

    def setstr(self, key, value, ex=None):
        try:
            key_bytes = key.encode('utf-8') if isinstance(key, str) else key
            value_bytes = value.encode('utf-8') if isinstance(value, str) else value
            
            # Store with expiration time if provided
            if ex is not None:
                expire_time = time.time() + ex
                data = {'value': value, 'expire_time': expire_time}
                value_bytes = pickle.dumps(data)
            
            with self.env.begin(write=True) as txn:
                txn.put(key_bytes, value_bytes)
            return True
        except:
            print(traceback.format_exc())
            return False
            
    def getstr(self, key):
        try:
            key_bytes = key.encode('utf-8') if isinstance(key, str) else key
            with self.env.begin() as txn:
                value_bytes = txn.get(key_bytes)
                if value_bytes is None:
                    return None
                
                # Try to unpickle to check if it has expiration
                try:
                    data = pickle.loads(value_bytes)
                    if isinstance(data, dict) and 'expire_time' in data:
                        if time.time() > data['expire_time']:
                            # Expired, delete and return None
                            with self.env.begin(write=True) as write_txn:
                                write_txn.delete(key_bytes)
                            return None
                        return data['value']
                except:
                    # Not pickled data, return as string
                    return value_bytes.decode('utf-8')
                
                return value_bytes.decode('utf-8')
        except:
            print(traceback.format_exc())
            return None
    
    def get_num_keys(self):
        with self.env.begin() as txn:
            return txn.stat()['entries']
    
    def incr(self, key):
        try:
            key_bytes = key.encode('utf-8') if isinstance(key, str) else key
            with self.env.begin(write=True) as txn:
                value_bytes = txn.get(key_bytes)
                if value_bytes is None:
                    current_val = 0
                else:
                    current_val = int(value_bytes.decode('utf-8'))
                new_val = current_val + 1
                txn.put(key_bytes, str(new_val).encode('utf-8'))
            return True
        except:
            print(traceback.format_exc())
            return False

    def mset(self, infos):
        try:
            with self.env.begin(write=True) as txn:
                for key, value in infos.items():
                    # For sets, we store as a pickled set
                    key_bytes = key.encode('utf-8')
                    existing = txn.get(key_bytes)
                    if existing:
                        try:
                            current_set = pickle.loads(existing)
                        except:
                            current_set = set()
                    else:
                        current_set = set()
                    
                    current_set.add(value)
                    txn.put(key_bytes, pickle.dumps(current_set))
            return True
        except:
            print(traceback.format_exc())
            return False

    def mget(self, keys):
        try:
            result = []
            with self.env.begin() as txn:
                for key in keys:
                    key_bytes = key.encode('utf-8')
                    value_bytes = txn.get(key_bytes)
                    if value_bytes:
                        try:
                            result.append(pickle.loads(value_bytes))
                        except:
                            result.append(set())
                    else:
                        result.append(set())
            return result
        except:
            print(traceback.format_exc())
            return False

    def set(self, key, value, ex=None):
        try:
            key_bytes = key.encode('utf-8')
            with self.env.begin(write=True) as txn:
                existing = txn.get(key_bytes)
                if existing:
                    try:
                        current_set = pickle.loads(existing)
                    except:
                        current_set = set()
                else:
                    current_set = set()
                
                was_new = value not in current_set
                current_set.add(value)
                
                # Handle expiration
                if ex is not None:
                    expire_time = time.time() + ex
                    data = {'value': current_set, 'expire_time': expire_time}
                    txn.put(key_bytes, pickle.dumps(data))
                else:
                    txn.put(key_bytes, pickle.dumps(current_set))
                
                return 1 if was_new else 0
        except:
            print(traceback.format_exc())
            return None
        
    def is_exist(self, key, value):
        try:
            key_bytes = key.encode('utf-8')
            with self.env.begin() as txn:
                value_bytes = txn.get(key_bytes)
                if value_bytes:
                    try:
                        current_set = pickle.loads(value_bytes)
                        return value in current_set
                    except:
                        return False
                return False
        except:
            print(traceback.format_exc())
            return 1

    def get(self, key):
        with self.env.begin() as txn:
            key_bytes = key.encode('utf-8')
            value_bytes = txn.get(key_bytes)
            if value_bytes:
                try:
                    return [pickle.loads(value_bytes)]
                except:
                    return [set()]
            return [set()]
    
    def remove(self, key, value):
        key_bytes = key.encode('utf-8')
        with self.env.begin(write=True) as txn:
            existing = txn.get(key_bytes)
            if existing:
                try:
                    current_set = pickle.loads(existing)
                    current_set.discard(value)
                    if current_set:
                        txn.put(key_bytes, pickle.dumps(current_set))
                    else:
                        txn.delete(key_bytes)
                except:
                    pass
    
    def mremove(self, infos):
        try:
            with self.env.begin(write=True) as txn:
                for key, value in infos.items():
                    key_bytes = key.encode('utf-8')
                    existing = txn.get(key_bytes)
                    if existing:
                        try:
                            current_set = pickle.loads(existing)
                            current_set.discard(value)
                            if current_set:
                                txn.put(key_bytes, pickle.dumps(current_set))
                            else:
                                txn.delete(key_bytes)
                        except:
                            pass
            return True
        except:
            print(traceback.format_exc())
            return False
    
    def delete(self, keys):
        with self.env.begin(write=True) as txn:
            for key in keys:
                key_bytes = key.encode('utf-8') if isinstance(key, str) else key
                txn.delete(key_bytes)
    
    def exists(self, key):
        try:
            key_bytes = key.encode('utf-8') if isinstance(key, str) else key
            with self.env.begin() as txn:
                return 1 if txn.get(key_bytes) is not None else 0
        except:
            print(traceback.format_exc())
            return 0
    
    def deleteallkeys(self):
        with self.env.begin(write=True) as txn:
            cursor = txn.cursor()
            keys_to_delete = [key for key, _ in cursor]
            for key in keys_to_delete:
                txn.delete(key)


class NoteInfoCacher(object):
    def __init__(self, db_path, prefix, ex):
        self.db_path = db_path
        self.prefix = prefix
        self.ex = ex
        self.lmdb_client = LMDBClient(db_path)

    def noteid_key(self, note_id):
        return f'{self.prefix}_noteid_{note_id}'

    def get(self, note_id):
        return self.lmdb_client.getstr(self.noteid_key(note_id))

    def get_note_info(self, note_id):
        info = self.get(note_id)
        if info is not None:
            info = json.loads(info)
            cached = 1
        else:
            info = None
        return info

    def cache_note_info(self, note_id, noteinfo):
        assert isinstance(noteinfo, dict)
        noteinfo_str = json.dumps(noteinfo)
        return self.lmdb_client.setstr(self.noteid_key(note_id), noteinfo_str, self.ex)

    def isCached(self, note_id):
        return 1 == self.lmdb_client.exists(self.noteid_key(note_id))
    

class UserInfoCacher(object):
    def __init__(self, db_path, prefix, ex):
        self.db_path = db_path
        self.prefix = prefix
        self.ex = ex
        self.lmdb_client = LMDBClient(db_path)

    def userid_key(self, user_id):
        return f'{self.prefix}_userid_{user_id}'

    def get(self, user_id):
        return self.lmdb_client.getstr(self.userid_key(user_id))

    def get_user_info(self, user_id):
        info = self.get(user_id)
        if info is not None:
            info = json.loads(info)
        else:
            info = None
        return info

    def cache_user_info(self, user_id, userinfo):
        assert isinstance(userinfo, dict)
        userinfo_str = json.dumps(userinfo)
        return self.lmdb_client.setstr(self.userid_key(user_id), userinfo_str, self.ex)
    
    def isCached(self, user_id):
        return 1 == self.lmdb_client.exists(self.userid_key(user_id))


class UserEngageLastnCacher(object):
    def __init__(self, db_path, prefix, ex):
        self.db_path = db_path
        self.prefix = prefix
        self.ex = ex
        self.lmdb_client = LMDBClient(db_path)

    def userid_key(self, user_id):
        return f'{self.prefix}_userid_{user_id}'

    def get(self, user_id):
        return self.lmdb_client.getstr(self.userid_key(user_id))

    def get_user_lastn_info(self, user_id):
        info = self.get(user_id)
        if info is not None:
            info = json.loads(info)
        else:
            info = None
        return info

    def cache_user_lastn_info(self, user_id, userinfo):
        userinfo_str = json.dumps(userinfo)
        return self.lmdb_client.setstr(self.userid_key(user_id), userinfo_str, self.ex)
    
    def isCached(self, user_id):
        return 1 == self.lmdb_client.exists(self.userid_key(user_id))
    

class ImageCache(object):
    def __init__(self, db_path, prefix, ex):
        self.lmdb_client = LMDBClient(db_path)
        self.prefix = prefix
        self.ex = ex
    
    def note_key(self, note_id):
        return f'{self.prefix}_{note_id}'

    def cache_img(self, note_id, pil_img, ex=1):
        img_byte_arr = BytesIO()
        pil_img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return self.lmdb_client.setstr(self.note_key(note_id), img_byte_arr, ex)
    
    def get(self, note_id):
        result = self.lmdb_client.getstr(self.note_key(note_id))
        if isinstance(result, str):
            return result.encode('utf-8')
        return result

    def get_pil_img(self, note_id):
        img_bin = self.get(note_id)
        if img_bin is None:
            return None
        return self.decode_img(img_bin)
    
    def isCached(self, note_id):
        return 1 == self.lmdb_client.exists(self.note_key(note_id))
    
    def decode_img(self, img_bin):
        return Image.open(BytesIO(img_bin)).convert('RGB')
