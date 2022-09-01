import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from scipy import signal
from PIL import Image
import pickle
import io
from google.cloud import storage


#TODO: enter values to the following variables & create directories TP9,TP10,AF7 & AF8
BUCKET_NAME = "brain-mnist"

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)


def create_single_spectro(X: pd.Series, local_img_path):
    """Plot & save img as np.array."""
    #create plot
    fig = plt.figure(frameon = False, dpi = 120)
    X_spectro = X.iloc[3:]
    f, t, Sxx = signal.stft(X_spectro.astype('float'))
    plt.pcolormesh(t, f, np.abs(Sxx), vmin=0, vmax=2, shading='gouraud')
    ax=plt.gca()
    ax.grid(visible=False)
    plt.box(on=False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    #Convert plot image into bytes
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=120)
    io_buf.seek(0)

    #Get array from bytes
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    plt.close()

    #Creating a source_file_name that will at first be upload locally the pickle file
    source_file_name=f'{local_img_path}{X.iloc[2]}/{X.iloc[0]}_{X.iloc[1]}.pickle'

    with open(source_file_name, 'wb') as handle:
        pickle.dump(img_arr[:,:,:-1], handle, protocol=pickle.HIGHEST_PROTOCOL)

    #Downloading from source_file_name and uploading to destination_blob_name_pickle
    destination_blob_name_pickle = f"gs://{BUCKET_NAME}/{X.iloc[2]}/{X.iloc[0]}_{X.iloc[1]}.pickle"

    upload_blob(BUCKET_NAME, source_file_name, destination_blob_name_pickle)

    #Return path of the created img saved as an array to fill the csv file comprising the signal values.
    return destination_blob_name_pickle


def add_arrays_to_pickle(X : pd.DataFrame, local_img_path):

    #Function to activate the file creation and to fill the column filepath of the signal csv
    X['file_path']=X.apply(lambda x:create_single_spectro(x, local_img_path),axis=1)

    return X

def add_arrays_to_pickle_by_chunk(CHUNK_SIZE,
                                  data_file_in_bucket,
                                  local_img_path,
                                  csv_path_in_bucket,
                                  chunk_id=0,
                                  chunk_max=0,
                                  to_bucket=True):

    #To save the array into a Google Storage bucket
    if to_bucket:
        while (True):

            #Early stop to avoid mass download
            print(f"download chunk n°{chunk_id}...")
            if chunk_max!=0 and chunk_id == chunk_max:
                print(f"download early stopped chunk n°{chunk_id}...")
                return None

            try:
                #Open csv with signal
                data_csv_source = f"gs://{BUCKET_NAME}/{data_file_in_bucket}"
                data_raw_chunk = pd.read_csv(
                        data_csv_source,
                        delimiter=',',
                        header=None,
                        skiprows=(chunk_id * CHUNK_SIZE) + 1,
                        nrows=CHUNK_SIZE
                        )

                #Replace y_true "-1" values into "10" for file naming
                data_raw_chunk = data_raw_chunk.replace({1: -1}, 10)

                #Save img and create dataframe with filepath column
                new_data_raw = add_arrays_to_pickle(data_raw_chunk,local_img_path)

                #Create or update csv file with filepath column
                destination_blob_name_csv = f"gs://{BUCKET_NAME}/{csv_path_in_bucket}"
                new_data_raw.to_csv(destination_blob_name_csv,
                    mode="w" if chunk_id==0 else "a",
                    header=chunk_id == 0,
                    index=False)

                upload_blob(BUCKET_NAME, data_csv_source, destination_blob_name_csv)

            #Handling errors
            except pd.errors.EmptyDataError:
                data_raw_chunk = None
            if data_raw_chunk is None:
                break
            if len(data_raw_chunk) == 0:
                break

            chunk_id += 1

        print(":white_check_mark: data saved entirely")

# #----------------------------------------------------------------------------
# #----------------------------------------------------------------------------
#     else :
#     #By default to save the array locally or in google drive
#         while (True):

#             #Early stop to avoid mass download
#             print(f"download chunk n°{chunk_id}...")
#             if early_stop and chunk_id == chunk_max:
#                 print(f"download early stopped chunk n°{chunk_id}...")
#                 return None

#             try:

#                 #Open csv with signal
#                 data_raw_chunk = pd.read_csv(
#                         data_path,
#                         delimiter=',',
#                         header=None,
#                         skiprows=(chunk_id * CHUNK_SIZE) + 1,
#                         nrows=CHUNK_SIZE
#                         )

#                 #Replace y_true "-1" values into "10" for file naming
#                 data_raw_chunk = data_raw_chunk.replace({1: -1}, 10)

#                 #Save img and create dataframe with filepath column
#                 new_data_raw = add_arrays_to_pickle(data_raw_chunk, local_img_path)

#                 #Create or update csv file with filepath column
#                 new_data_raw.to_csv(csv_path,
#                     mode="w" if chunk_id==0 else "a",
#                     header=chunk_id == 0,
#                     index=False)

#             #Handling errors
#             except pd.errors.EmptyDataError:
#                 data_raw_chunk = None
#             if data_raw_chunk is None:
#                 break
#             if len(data_raw_chunk) == 0:
#                 break

#             chunk_id += 1

#         print(":white_check_mark: data saved entirely")


if __name__ == '__main__':
    CHUNK_SIZE=5
    data_file_in_bucket='M2_clean.txt'
    local_img_path='data/images/'
    csv_path_in_bucket='M2_path.csv'
    add_arrays_to_pickle_by_chunk(CHUNK_SIZE,
                                  data_file_in_bucket,
                                  local_img_path,
                                  csv_path_in_bucket,
                                  chunk_id=0,
                                  chunk_max=2,
                                  to_bucket=True)
