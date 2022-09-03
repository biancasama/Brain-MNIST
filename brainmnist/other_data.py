import pandas as pd
import numpy as np
from data import balance_data, map_data_array3D


def load_other_data() -> pd.DataFrame:
    """
    load other data from txt format
    """
    BUCKET_NAME = "brain-mnist"
    data = pd.read_csv(f"gs://{BUCKET_NAME}/other_datasets/MU.txt", sep='\t', header=None)
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

    data = data.sort_values(by='index_event').iloc[:40,:]

    #function to take into account different number of data point for each event
    def func_apply(x):
        try:
            return int(x.str.split(',').iloc[0][i])

        except IndexError:
            return np.nan

    #dispatch eeg signals in multiple columns
    for i in range(max_data_points):

        #print advancement as long command...
        if i%100==0: print(f'Data point {i+1} out of {max_data_points}')

        #select i-th data point in eeg signal and put it in a new column
        point_i_eeg = pd.DataFrame(data.iloc[:,3]).apply(func_apply, axis=1)
        data = pd.concat([data, point_i_eeg], axis=1)
        data.columns = list(data.columns[:-1]) + [i]

    #drop former eeg signal
    data = data.drop(columns='eeg')

    #save in bucket
    BUCKET_NAME = "brain-mnist"
    data.to_csv(f"gs://{BUCKET_NAME}/other_datasets/MU_clean.csv")

    return data



if __name__=='__main__':
    df = load_other_data()
    df = map_other_data(df)
    print(df.shape)
    print(df.head())

    # BUCKET_NAME = "brain-mnist"
    # df = pd.read_csv(f"gs://{BUCKET_NAME}/other_datasets/MU_clean.txt", sep='\t', header=None)
    # df = balance_data(df)
    # X, y = map_data_array3D(df)
    # print(X.shape)
    # print(y.shape)
