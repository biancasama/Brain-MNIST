import pandas as pd

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
  df_base = data_.loc[:,['true_digit','index_event']] #keep only true_digit
  df1 = data_.iloc[:,(1+1):(1+1+512)] #keep only channel 1
  df2 = data_.iloc[:,(1+1+512):(1+1+512*2)] #keep only channel 2
  df3 = data_.iloc[:,(1+1+512*2):(1+1+512*3)] #keep only channel 3
  df4 = data_.iloc[:,(1+1+512*3):(1+1+512*4)] #keep only channel 4
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

  #concatenate datasets for all channels
  df = pd.concat([df1_m, df2_m, df3_m, df4_m])
  del df1_m, df2_m, df3_m, df4_m

  #re-arrange order of columns
  column_names = ['index_event', 'true_digit', 'channel'] + list(range(1,513))
  df = df.reindex(columns=column_names)

  return df
