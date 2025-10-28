import json
import os
import pickle

def delivery_report(err, msg):
    if err is not None:
        print('error', 'message delivery failed: {}'.format(msg), str(err))

def save_local(embeddings_data, output_file="embeddings"):
    # Check if directory exists, create it if not
    directory = os.path.dirname(output_file)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    # Save as Pickle (binary format)
    with open(output_file, "wb") as f:
        pickle.dump(embeddings_data, f)