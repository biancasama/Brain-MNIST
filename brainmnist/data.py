import pandas as pd
import numpy as np


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

    #number of representations of the digit the less represented for each electrode (TP9 is chosen arbitrarily, no impact)
    min_count_digits = df[df.channel=='TP9'].groupby('true_digit').count().iloc[:,0].min()

    df_concat = pd.DataFrame()

    #sample min_count_digits different index_event for true_digit from -1 to 9
    for i in range(-1,10):

        #identify all events related to the true_digit considered
        events_of_digit = df[df.channel=='TP9'][df[df.channel=='TP9'].true_digit==i]['index_event']

        #sample min_count_digits events among all events related to true_digit considered
        sample_events_of_digit = events_of_digit.sample(min_count_digits,replace=False).tolist()

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

        #extract eeg data (of 4 channels) related to a specific index_event a put then in ilst of list format
        eeg_index_event = df[df.index_event==df.index_event.unique()[i]].drop(columns=['index_event','true_digit','channel']).T.values.tolist()
        #concatenate eeg data coming from all events
        X_list.append(eeg_index_event)

        #extract y data related to a specific index_event & concatenate them
        y_list.append(df[df.index_event==df.index_event.unique()[i]]['true_digit'].tolist()[0])


    X = np.array(X_list)
    y = np.array(y_list)
    del X_list, y_list

    return X, y



if __name__=='__main__':
    df = load_clean_data_from_bucket()
    df = balance_data(df)
    print(df.shape)
-
