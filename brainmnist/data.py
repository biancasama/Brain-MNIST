import pandas as pd
import numpy as np
from scipy import signal
from brainmnist.filtering import notch_filter, butter_bandpass_filter
from cloud_data import upload_blob
from google.cloud import storage


def load_data() -> pd.DataFrame:
    """
    load data from txt format
    """
    data = pd.read_csv('drive/MyDrive/Brain/data/MU2.txt', sep=',', header=None)
    return data



def map_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    map data in relevant format:
    keep event_index, true_digit, channel & EEG signal
    """

    #delete useless last variables (PPGdata, Accdata, Gyrodata)
    data_ = data.iloc[:,:(784+3+1+512*4)]
    del data

    #create a variable 'index_event'
    data_['index_event'] = range(1,data_.shape[0]+1)

    #save images-related data in another dataset
    MU_images = data_.iloc[:,[1] + list(data_.iloc[:,3:(784+3)].columns) + [data_.columns.get_loc('index_event')]]

    #delete useless variables for CNN (TRAIN/TEST as only TRAIN, MNIST origin, timestamp and image pixels)
    data_ = data_.drop(([0,1,787]+list(data_.iloc[:,3:(784+3)].columns)),axis=1)

    #rename first variable as true_digit
    data_ = data_.rename(columns = {2: 'true_digit'})

    ##mapping of EEG data

    #create intermediary datasets
    df_base = data_.loc[:,['true_digit','index_event']] #keep only true_digit and index_event
    df1 = data_.iloc[:,(1):(1+512)] #keep only channel 1
    df2 = data_.iloc[:,(1+512):(1+512*2)] #keep only channel 2
    df3 = data_.iloc[:,(1+512*2):(1+512*3)] #keep only channel 3
    df4 = data_.iloc[:,(1+512*3):(1+512*4)] #keep only channel 4
    del data_

    #add channel information
    df1['channel'] = 'TP9'
    df2['channel'] = 'AF7'
    df3['channel'] = 'AF8'
    df4['channel'] = 'TP10'

    #merge with true_digit
    df1_m = df_base.merge(df1,left_index=True, right_index=True)
    df2_m = df_base.merge(df2,left_index=True, right_index=True)
    df3_m = df_base.merge(df3,left_index=True, right_index=True)
    df4_m = df_base.merge(df4,left_index=True, right_index=True)

    #set same name of columns for all channels (from 1 to 512)
    df1_m.columns =  ['true_digit','index_event'] + list(range(1,512+1)) + ['channel']
    df2_m.columns =  ['true_digit','index_event'] + list(range(1,512+1)) + ['channel']
    df3_m.columns =  ['true_digit','index_event'] + list(range(1,512+1)) + ['channel']
    df4_m.columns =  ['true_digit','index_event'] + list(range(1,512+1)) + ['channel']

    df = pd.concat([df1_m, df2_m, df3_m, df4_m])
    del df1_m, df2_m, df3_m, df4_m

    column_names = ['index_event', 'true_digit', 'channel'] + list(range(1,513))
    df = df.reindex(columns=column_names)

    return df



def load_clean_data_from_bucket() -> pd.DataFrame:
    """
    Load clean data (ie mapped) from bucket
    """

    BUCKET_NAME = "brain-mnist"
    data_file_in_bucket = 'MU2_clean.csv'
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{data_file_in_bucket}")

    return df



def balance_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balance data acoording to the less represented digit
    """

    #number of representations of the digit the less represented for each electrode (first channel is chosen arbitrarily, no impact)
    min_count_digits = df[df.channel==df.channel[0]].groupby('true_digit').count().iloc[1:,0].min()

    df_concat = pd.DataFrame()

    #sample min_count_digits different index_event for true_digit from -1 to 9
    for i in range(-1,10):

        #identify all events related to the true_digit considered
        events_of_digit = df[df.channel==df.channel[0]][df[df.channel==df.channel[0]].true_digit==i]['index_event']

        #sample min_count_digits events among all events related to true_digit considered
        if len(events_of_digit) > min_count_digits:
            sample_events_of_digit = events_of_digit.sample(min_count_digits,replace=False).tolist()
        else:
            sample_events_of_digit = events_of_digit.tolist()

        #keep only events that were sampled for the digit
        df_sample_digit = df[df.index_event.isin(sample_events_of_digit)]

        #concatenate events that were sampled for all  digits
        df_concat = pd.concat([df_concat, df_sample_digit])

    return df_concat



def map_data_array3D(df: pd.DataFrame) -> tuple:
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

    return X, y



def map_filtered_data_array3D(df: pd.DataFrame) -> tuple:
    """
    Map other data in a 3-dimensional array (nb_seq,nb_obs,n_features)=(nb_seq,xx,4)
    nb_seq depend on the data used as input (full dataset or balanced dataset)
    xx depends on event
    """

    X_list=[]
    y_list=[]

    for i in range(len(df.index_event.unique())):
        #extract eeg data (of 4 channels) related to a specific index_event a put them in list of list format
        eeg_index_event = [list(e) for e in df[df.index_event==df.index_event.unique()[0]].drop(columns=['index_event','true_digit','channel'])['eeg'].T.to_list()]
        eeg_index_event_manip = np.array(eeg_index_event).T.tolist()
        #concatenate eeg data coming from all events
        X_list.append(eeg_index_event_manip)

        #extract y data related to a specific index_event & concatenate them
        y_list.append(df[df.index_event==df.index_event.unique()[i]]['true_digit'].tolist()[0])


    X = np.array(X_list).astype('float32')
    y = np.array(y_list)
    del X_list, y_list

    ##save X and y as blobs in bucket
    BUCKET_NAME = "brain-mnist"
    np.save(f'data/MU2_clean_X.npy', X, allow_pickle=True, fix_imports=True) #save X locally
    np.save(f'data/MU2_clean_y.npy', y, allow_pickle=True, fix_imports=True) #save y locally
    upload_blob(BUCKET_NAME, f'data/MU2_clean_X.npy', f"MU2_clean_X.npy")
    upload_blob(BUCKET_NAME, f'data/MU2_clean_y.npy', f"MU2_clean_y.npy")

    return X, y




def filtering(X: pd.Series):

    #extract signal
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

    return sample_butter



def download_blob(bucket_name, source_file_name, destination_blob_name):
    """Download a file from the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.download_to_filename(source_file_name)



if __name__=='__main__':

    # BUCKET_NAME = "brain-mnist"
    # df = pd.read_csv(f"gs://{BUCKET_NAME}/MU2_clean.csv")

    # df = balance_data(df)

    # X, y = map_data_FT_array4D(df)
    # print(X.shape)
    # print(len(X), len(X[0]), len(X[0][0]), len(X[0][0][0]))
    # print(y.shape)

    images = load_blob_images()
    print(images.shape)
