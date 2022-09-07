import pandas as pd
import numpy as np
from scipy import signal
from brainmnist.filtering import notch_filter, butter_bandpass_filter
from cloud_data import upload_blob
from google.cloud import storage


def plot_ERPs(df: pd.DataFrame, sf, chan_list: list):
    """Import dataframe of filtered data, compute time axis, mean, standard error of the mean
    and plot Confidence boundaries around the mean signal in a shaded version"""
    fig, axs = plt.subplots(5, 2, figsize=(18, 12))

    dig= 0

    for ax in axs.ravel():

    tmp= butterMU2[butterMU2['true_digit']== dig]

    for chan in chan_list:

        df= tmp[tmp['channel']== chan]

        t = np.arange(df.iloc[:,3:].shape[1]) / sf
        ntrials = df.iloc[:,3:].shape[1]
        mn= df.iloc[:,3:].mean(axis=0)
        sd= df.iloc[:,3:].std(axis=0)
        sem = sd / np.sqrt(ntrials)

        ax.plot(t, mn, lw=1.5, label= f'Channel {chan}')
        ax.fill_between(t, mn - 2*sem, mn + 2*sem, alpha=0.2) # plot upper, lower CI

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage ($\mu$ V)')
        handles, labels= ax.get_legend_handles_labels()
        ax.set_xlim([t.min(), t.max()])
        ax.set_title(f'ERP of Digit {dig}')

    dig += 1

    fig.legend(handles, labels, loc= 'upper right')
    plt.tight_layout();
    return fig

BUCKET_NAME = "brain-mnist"
data_file_in_bucket = 'MU2_clean.csv'
df = pd.read_csv(f"gs://{BUCKET_NAME}/{data_file_in_bucket}")
fig = plot_ERPs(df, sf, chan_list)
fig.savefig("results/ERPs.png")
