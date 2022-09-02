import tensorflow as tf
import numpy as np
import os

BUCKET_NAME = "brain-mnist"


def get_data_EEG(EEG_type):
    try :
        img_directory = f"gs://{BUCKET_NAME}/{EEG_type}"
        img_list = os.list_dir(img_directory)

        dataset = tf.data.Dataset.from_tensor_slices(img_list)

        dataset = dataset.map(lambda item: tf.numpy_function(np.load, [item], tf.unit8),
                            num_parallel_calls=tf.data.AUTOTUNE)

    except:
        raise ValueError('Please enter an existing EEG file: TP9,TP10, AF7, AF8')

def initialize_mode():
    #TODO : Complete model architecture

def compile():
    #TODO : Complete model compilation

def train():
    #TODO : Complete model training
