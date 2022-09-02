from package.filtering import notch_filter, butter_bandpass_filter
from skimage.transform import resize
from google.cloud import storage
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import numpy as np
import pickle
import io
import os


# import tensorflow as tf
# from PIL import Image


#TODO: enter values to the following variables & create directories TP9,TP10,AF7 & AF8
BUCKET_NAME = "brain-mnist"

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

def create_single_spectro(X: pd.Series, local_img_path,remove_local=False,img_format='npy'):
    """Function to generate a plot & img as np.array from a signal on bucket"""

    #extract signal
    fig = plt.figure(frameon = False, dpi = 120)
    X_spectro = X.iloc[3:]

    #filter signal
    fs = 256
    Q = 25
    w0 = 50
    lowcut = 14
    highcut = 71
    order = 6
    sample_notch = notch_filter(X_spectro, w0, Q, fs)
    sample_butter = butter_bandpass_filter(sample_notch, lowcut, highcut, fs, order)

    #create plot
    f, t, Sxx = signal.stft(sample_butter.astype('float'))
    plt.pcolormesh(t, f, np.abs(Sxx), vmin=0, vmax=2, shading='gouraud')

    #keeping just the image
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
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(),dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))

    #resize array by dividing shape by 3 and selecting unit8 as dtype to save space
    img_arr = img_arr[:,:,:-1]
    img_arr = resize(img_arr, (img_arr.shape[0]/3,img_arr.shape[1]/3)) * 255
    img_arr = img_arr.astype('uint8')

    #closing bytes-object and plt
    io_buf.close()
    plt.close()

    #Creating a source_file_name that will at first be upload locally the npy file
    if img_format=='npy':
        #alternatives paths TO_BE_VALIDATED to avoid directory creation and simplify removing task
        #file_name = {X.iloc[0]}_{X.iloc[1]}.npy'
        #source_file_name=f'{local_img_path}{file_name}'
        #destination_file_name=f'{local_img_path}{X.iloc[2]}/{file_name}'

        source_file_name=f'{local_img_path}{X.iloc[2]}/{X.iloc[0]}_{X.iloc[1]}.npy'
        np.save(source_file_name, img_arr, allow_pickle=True, fix_imports=True)

        #Downloading from source_file_name and uploading to destination_blob_name_pickle
        destination_blob_name_pickle = f"{X.iloc[2]}/{X.iloc[0]}_{X.iloc[1]}.npy"

        upload_blob(BUCKET_NAME, source_file_name, destination_blob_name_pickle)

        #remove file after usage
        if remove_local:
            os.remove(source_file_name)

        #Return path of the created img saved as an array to fill the csv file comprising the signal values.
        return destination_blob_name_pickle

    elif img_format=='pickle':
        #alternatives paths TO_BE_VALIDATED to avoid directory creation and simplify removing task
        #file_name = {X.iloc[0]}_{X.iloc[1]}.pickle'
        #source_file_name=f'{local_img_path}{file_name}'
        #destination_file_name=f'{local_img_path}{X.iloc[2]}/{file_name}'

        source_file_name=f'{local_img_path}{X.iloc[2]}/{X.iloc[0]}_{X.iloc[1]}.pickle'

        with open(source_file_name, 'wb') as handle:
            pickle.dump(img_arr[:,:,:-1], handle, protocol=pickle.HIGHEST_PROTOCOL)

        #Downloading from source_file_name and uploading to destination_blob_name_pickle
        destination_blob_name_pickle = f"gs://{BUCKET_NAME}/{X.iloc[2]}/{X.iloc[0]}_{X.iloc[1]}.pickle"

        upload_blob(BUCKET_NAME, source_file_name, destination_blob_name_pickle)

        #remove file after usage
        if remove_local:
            os.remove(source_file_name)

        #Return path of the created img saved as an array to fill the csv file comprising the signal values.
        return destination_blob_name_pickle

    raise ValueError('Please select correct img_format: npy or pickle')


def add_arrays_to_pickle(X : pd.DataFrame, local_img_path):
    '''Function to activate the file creation and to fill the column filepath of the signal csv'''

    X['file_path']=X.apply(lambda x:create_single_spectro(x, local_img_path),axis=1)

    return X

def add_arrays_to_pickle_by_chunk(CHUNK_SIZE,
                                  data_file_in_bucket,
                                  local_img_path,
                                  csv_path_in_bucket,
                                  chunk_id=0,
                                  chunk_max=0,
                                  to_bucket=True):
    '''From a DataFrame, upload images to bucket chunk by chunk'''

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
                # data = gcp_csv_to_df(BUCKET_NAME, data_file_in_bucket)
                data_raw_chunk = pd.read_csv(
                        # io.BytesIO(data),
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
                # local_csv = 'data/new.csv'
                new_data_raw.to_csv(destination_blob_name_csv,
                    mode="w" if chunk_id==0 else "a",
                    header=chunk_id == 0,
                    index=False)

                # upload_blob(BUCKET_NAME, local_csv, destination_blob_name_csv)

            #Handling errors
            except pd.errors.EmptyDataError:
                data_raw_chunk = None
            if data_raw_chunk is None:
                break
            if len(data_raw_chunk) == 0:
                break

            chunk_id += 1

        print(":white_check_mark: data saved entirely")

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
    else :
    #By default to save the array locally or in google drive
        while (True):

            #Early stop to avoid mass download
            print(f"download chunk n°{chunk_id}...")
            if chunk_max and chunk_id == chunk_max:
                print(f"download early stopped chunk n°{chunk_id}...")
                return None

            try:

                #Open csv with signal
                data_raw_chunk = pd.read_csv(
                        data_file_in_bucket,
                        delimiter=',',
                        header=None,
                        skiprows=(chunk_id * CHUNK_SIZE) + 1,
                        nrows=CHUNK_SIZE
                        )

                #Replace y_true "-1" values into "10" for file naming
                data_raw_chunk = data_raw_chunk.replace({1: -1}, 10)

                #Save img and create dataframe with filepath column
                new_data_raw = add_arrays_to_pickle(data_raw_chunk, local_img_path)

                #Create or update csv file with filepath column
                new_data_raw.to_csv(csv_path_in_bucket,
                    mode="w" if chunk_id==0 else "a",
                    header=chunk_id == 0,
                    index=False)

            #Handling errors
            except pd.errors.EmptyDataError:
                data_raw_chunk = None
            if data_raw_chunk is None:
                break
            if len(data_raw_chunk) == 0:
                break

            chunk_id += 1

        print(":white_check_mark: data saved entirely")


if __name__ == '__main__':
    add_arrays_to_pickle_by_chunk(CHUNK_SIZE=5,
                                  data_file_in_bucket='MU2_clean.txt',
                                  local_img_path='data/images/',
                                  csv_path_in_bucket='MU2_path.csv',
                                  chunk_id=0,
                                  chunk_max=2,
                                  to_bucket=True)
