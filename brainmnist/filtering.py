from scipy.signal import butter, lfilter, iirnotch, stft
import pandas as pd
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def notch_filter(data, w0, Q, fs):
    b, a = iirnotch(w0, Q, fs)
    y = lfilter(b, a, data)
    return y
