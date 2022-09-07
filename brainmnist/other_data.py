import pandas as pd
import numpy as np
from data import balance_data
from cloud_data import upload_blob
from google.cloud import storage
from filtering import notch_filter, butter_bandpass_filter


def load_other_data() -> pd.DataFrame:
    """
    load other data from txt format
    """
    BUCKET_NAME = "brain-mnist"
    data = pd.read_csv(f"gs://{BUCKET_NAME}/other_datasets/{dataset_name}.txt", sep='\t', header=None)
    return data


def map_other_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    map other data in relevant format:
    keep event_index, true_digit, channel & EEG signal
    """

    # min_data_points = data.iloc[:,5].min()
    max_data_points = data.iloc[:,5].max()

    data = data.drop(columns=[0,2,5]) #drop useless columns

    data.columns = ['index_event', 'channel', 'true_digit', 'eeg'] #rename columns
    data = data.reindex(columns=['index_event', 'true_digit', 'channel', 'eeg']) #reorder columns

    data = pd.concat([data.iloc[:,:3], pd.DataFrame(data.iloc[:,3].apply(lambda x: x.split(',')))], axis=1)

    data1 = pd.DataFrame(data.eeg.values.tolist())
    data = pd.concat([data.drop('eeg',axis=1), data1], axis=1)

    #save in bucket
    BUCKET_NAME = "brain-mnist"
    data.to_csv(f'gs://{BUCKET_NAME}/other_datasets/{dataset_name}_clean.csv', index=False)

    return data


# def map_other_data(data: pd.DataFrame) -> pd.DataFrame:
#     """
#     map other data in relevant format:
#     keep event_index, true_digit, channel & EEG signal
#     """

#     # min_data_points = data.iloc[:,5].min()
#     max_data_points = data.iloc[:,5].max()

#     data = data.drop(columns=[0,2,5]) #drop useless columns

#     data.columns = ['index_event', 'channel', 'true_digit', 'eeg'] #rename columns
#     data = data.reindex(columns=['index_event', 'true_digit', 'channel', 'eeg']) #reorder columns

#     concat = pd.DataFrame()
#     for i in range(data.shape[0]):
#       if i%100==0: print('Row ' + str(i) + '/' + str(data.shape[0]))
#       concat = pd.concat([concat, pd.DataFrame(data.loc[i,'eeg'].split(',')).T], axis=0)
#     concat.index = list(range(0,data.shape[0]))

#     data = pd.concat([data.iloc[:,:3], concat], axis=1)

#     #save in bucket
#     BUCKET_NAME = "brain-mnist"
#     data.to_csv(f'gs://{BUCKET_NAME}/other_datasets/{dataset_name}_clean.csv', index=False)

#     return data



def map_other_data_array3D(df: pd.DataFrame) -> tuple:
    """
    Map data in a 3-dimensional array (nb_seq,nb_obs,n_features)=(nb_seq,512,4)
    nb_seq depend on the data used as input (full dataset or balanced dataset)
    """

    X_list=[]
    y_list=[]

    for i in range(len(df.index_event.unique())):

        #extract eeg data (of 4 channels) related to a specific index_event a put them in list of list format
        eeg_index_event = df[df.index_event==df.index_event.unique()[i]].drop(columns=['index_event','true_digit','channel']).T.values.tolist()
        #concatenate eeg data coming from all events
        X_list.append(eeg_index_event)

        #extract y data related to a specific index_event & concatenate them
        y_list.append(df[df.index_event==df.index_event.unique()[i]]['true_digit'].tolist()[0])


    X = np.array(X_list)
    y = np.array(y_list)
    del X_list, y_list

    ##save X and y as blobs in bucket
    BUCKET_NAME = "brain-mnist"
    np.save(f'data/{dataset_name}_filtered_X.npy', X, allow_pickle=True, fix_imports=True) #save X locally
    np.save(f'data/{dataset_name}_filtered_y.npy', y, allow_pickle=True, fix_imports=True) #save y locally
    upload_blob(BUCKET_NAME, f'data/{dataset_name}_filtered_X.npy', f"other_datasets/{dataset_name}_filtered_X.npy")
    upload_blob(BUCKET_NAME, f'data/{dataset_name}_filtered_y.npy', f"other_datasets/{dataset_name}_filtered_y.npy")

    return X, y



def download_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.download_to_filename(source_file_name)



# def map_other_data_array3D(df: pd.DataFrame) -> tuple:
#     """
#     Map other data in a 3-dimensional array (nb_seq,nb_obs,n_features)=(nb_seq,xx,4)
#     nb_seq depend on the data used as input (full dataset or balanced dataset)
#     xx depends on event
#     """

#     #put eeg data in list format (it was saved as a string in csv)
#     eeg_in_list = pd.DataFrame(df.loc[:,'eeg']).apply(lambda x: [float(e) for e in x.str.split(',').iloc[0]], axis=1)
#     df = pd.concat([df, eeg_in_list], axis=1)
#     df = df.drop(columns=['eeg'])
#     df.columns = ['index_event', 'true_digit', 'channel', 'eeg'] #rename columns

#     X_list=[]
#     y_list=[]

#     for i in range(len(df.index_event.unique())):

#         #extract eeg data (of 4 channels) related to a specific index_event a put them in list of list format
#         eeg_index_event = df[df.index_event==df.index_event.unique()[i]].drop(columns=['index_event','true_digit','channel']).values.tolist()
#         eeg_index_event_manip = np.array([e[0] for e in [el for el in eeg_index_event]]).T.tolist()
#         #concatenate eeg data coming from all events
#         X_list.append(eeg_index_event_manip)

#         #extract y data related to a specific index_event & concatenate them
#         y_list.append(df[df.index_event==df.index_event.unique()[i]]['true_digit'].tolist()[0])


#     X = np.array(X_list,dtype=object) #specify dtype=object to allow different nb of length of sequences
#     y = np.array(y_list)
#     del X_list, y_list

#     ##save X and y as blobs in bucket
#     BUCKET_NAME = "brain-mnist"
#     np.save(f'data/{dataset_name}_clean_X.npy', X, allow_pickle=True, fix_imports=True) #save X locally
#     np.save(f'data/{dataset_name}_clean_y.npy', y, allow_pickle=True, fix_imports=True) #save y locally
#     upload_blob(BUCKET_NAME, f'data/{dataset_name}_clean_X.npy', f"other_datasets/{dataset_name}_clean_X.npy")
#     upload_blob(BUCKET_NAME, f'data/{dataset_name}_clean_y.npy', f"other_datasets/{dataset_name}_clean_y.npy")

#     return X, y





if __name__=='__main__':

    dataset_name = 'EP1.01'

    # df = load_other_data()
    # print(df.shape)
    # df = map_other_data(df)
    # print(df.shape)

    BUCKET_NAME = "brain-mnist"
    df = pd.read_csv(f"gs://{BUCKET_NAME}/other_datasets/{dataset_name}_clean.csv")
    print(df.shape)
    print(df.head())

    df = balance_data(df)
    #delete columns only composed of nans (possible if subset of data)
    try:
        tmp = df.isnull().sum(axis=0)
        cols_nan = int(tmp[tmp==df.shape[0]].index[0])
        df = df.iloc[:,:(3+cols_nan)]
    except:
        pass
    print(df.shape)
    print(df.head())

    #filtering
    fs = 256 #sampling rate
    lowcut = 14 #high-pass filter of sufficient frequency to remove DC offset
    highcut = 70
    notch = pd.concat([df.iloc[:,:3].reset_index().drop(columns='index'),
                       pd.DataFrame(notch_filter(df.iloc[:,3:], 50, 25, fs))], axis= 1)
    #delete columns only composed of nans (possible if subset of data)
    tmp = notch.isnull().sum(axis=0)
    try:
        cols_nan = int(tmp[tmp==notch.shape[0]].index[0])
        notch = notch.iloc[:,:(3+cols_nan)]
    except:
        pass
    print(notch.shape)
    print(notch.head())

    df_butter = pd.concat([notch.iloc[:,:3].reset_index().drop(columns='index'),
                           pd.DataFrame(butter_bandpass_filter(notch.iloc[:,3:], lowcut, highcut, fs, order=2))], axis= 1)
    df_butter = df_butter.replace({np.nan: -1000}) #replace nans by 1000 for paading to come
    tmp = df_butter.isnull().sum(axis=0)
    try:
        cols_nan = int(tmp[tmp==df_butter.shape[0]].index[0])
        df_butter = df_butter.iloc[:,:(3+cols_nan)]
    except:
        pass
    print(df_butter.shape)
    print(df_butter.head())

    X, y = map_other_data_array3D(df_butter)
    print(X.shape)
    print(len(X), len(X[0]), len(X[0][0]))
    print(len(X), len(X[1]), len(X[0][0]))
    print(y.shape)
