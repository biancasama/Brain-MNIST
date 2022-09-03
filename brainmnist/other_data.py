import pandas as pd
import numpy as np


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

    #dispatch eeg signals in multiple columns
    for i in range(max_data_points):
        data = pd.concat([data, pd.DataFrame(data.iloc[:,3]).apply(lambda x: int(x.str.split(',').iloc[0][i]), axis=1)], axis=1)
        data.columns = list(data.columns[:-1]) + [i]
    data = data.drop(columns='eeg')

    return data



if __name__=='__main__':
    df = load_other_data()
    df = df.sort_values(by='index_event').iloc[:20000,:]
    df = map_other_data(df)
    # df = balance_data(df)
    # X, y = map_data_array3D(df)
    # print(X.shape)
    # print(y.shape)
    print(df.shape)
    print(df.head())
