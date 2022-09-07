
# from other_data import download_blob
dataset_name='EP1.01'
import numpy as np
from google.cloud import storage

def download_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.download_to_filename(source_file_name)

BUCKET_NAME = "brain-mnist"
download_blob(BUCKET_NAME, f'data/{dataset_name}_filtered_X.npy', f"other_datasets/{dataset_name}_filtered_X.npy")
download_blob(BUCKET_NAME, f'data/{dataset_name}_filtered_y.npy', f"other_datasets/{dataset_name}_filtered_y.npy")
X = np.load(f'data/{dataset_name}_filtered_X.npy', allow_pickle=True, fix_imports=True)
y = np.load(f'data/{dataset_name}_filtered_y.npy', allow_pickle=True, fix_imports=True)

print(X.shape)
print(len(X), len(X[0]), len(X[0][0]))
print(len(X), len(X[1]), len(X[0][0]))
print(y.shape)
