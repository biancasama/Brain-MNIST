import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping

import pathlib
import os

import numpy as np

import matplotlib.pyplot as plt

#copy all data from the bucket on the VM (one time)
# #execute in the VM terminal at the root BRAIN-MNIST:

# #create folders:
# mkdir data/images/zero
# mkdir data/images/one
# mkdir data/images/two
# mkdir data/images/three
# mkdir data/images/four
# mkdir data/images/five
# mkdir data/images/six
# mkdir data/images/seven
# mkdir data/images/eight
# mkdir data/images/nine
# mkdir data/images/nothing

# #copy files from bucket
# gsutil -m cp gs://brain-mnist/data/zero/\*.npy data/images/zero
# gsutil -m cp gs://brain-mnist/data/one/\*.npy data/images/one
# gsutil -m cp gs://brain-mnist/data/two/\*.npy data/images/two
# gsutil -m cp gs://brain-mnist/data/three/\*.npy data/images/three
# gsutil -m cp gs://brain-mnist/data/four/\*.npy data/images/four
# gsutil -m cp gs://brain-mnist/data/five/\*.npy data/images/five
# gsutil -m cp gs://brain-mnist/data/six/\*.npy data/images/six
# gsutil -m cp gs://brain-mnist/data/seven/\*.npy data/images/seven
# gsutil -m cp gs://brain-mnist/data/eight/\*.npy data/images/eight
# gsutil -m cp gs://brain-mnist/data/nine/\*.npy data/images/nine

#copy only a subset of -1 to have approximately 8000 of them (approx. 2000 by channel)
# (events beginning by 4, 5, 6 or 7 are chosen arbitrarily to have the wanted nb of events)
# gsutil -m cp gs://brain-mnist/data/nothing/TP9_4\*.npy data/images/nothing
# gsutil -m cp gs://brain-mnist/data/nothing/TP9_5\*.npy data/images/nothing
# gsutil -m cp gs://brain-mnist/data/nothing/TP9_6\*.npy data/images/nothing
# gsutil -m cp gs://brain-mnist/data/nothing/TP9_7\*.npy data/images/nothing
# gsutil -m cp gs://brain-mnist/data/nothing/TP10_4\*.npy data/images/nothing
# gsutil -m cp gs://brain-mnist/data/nothing/TP10_5\*.npy data/images/nothing
# gsutil -m cp gs://brain-mnist/data/nothing/TP10_6\*.npy data/images/nothing
# gsutil -m cp gs://brain-mnist/data/nothing/TP10_7\*.npy data/images/nothing
# gsutil -m cp gs://brain-mnist/data/nothing/AF7_4\*.npy data/images/nothing
# gsutil -m cp gs://brain-mnist/data/nothing/AF7_5\*.npy data/images/nothing
# gsutil -m cp gs://brain-mnist/data/nothing/AF7_6\*.npy data/images/nothing
# gsutil -m cp gs://brain-mnist/data/nothing/AF7_7\*.npy data/images/nothing
# gsutil -m cp gs://brain-mnist/data/nothing/AF8_4\*.npy data/images/nothing
# gsutil -m cp gs://brain-mnist/data/nothing/AF8_5\*.npy data/images/nothing
# gsutil -m cp gs://brain-mnist/data/nothing/AF8_6\*.npy data/images/nothing
# gsutil -m cp gs://brain-mnist/data/nothing/AF8_7\*.npy data/images/nothing

# #check:
# cd data/images/zero ; ls | wc -l ; cd ../../..
# cd data/images/one ; ls | wc -l ; cd ../../..
# cd data/images/two ; ls | wc -l ; cd ../../..
# cd data/images/three ; ls | wc -l ; cd ../../..
# cd data/images/four ; ls | wc -l ; cd ../../..
# cd data/images/five ; ls | wc -l ; cd ../../..
# cd data/images/six ; ls | wc -l ; cd ../../..
# cd data/images/seven ; ls | wc -l ; cd ../../..
# cd data/images/eight ; ls | wc -l ; cd ../../..
# cd data/images/nine ; ls | wc -l ; cd ../../..
# cd data/images/nothing ; ls | wc -l ; cd ../../..

# data_dir = pathlib.Path(f'{os.getenv("HOME")}/code/fla66/Brain-MNIST/data/images')
data_dir = pathlib.Path(f'data/images')
print(data_dir)

image_count = len(list(data_dir.glob('*/*.npy')))
print(image_count)

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.npy'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != ".ipynb_checkpoints"]))

CONFIG = dict(test_split=.3,
              batch_size=32,
              img_height = 192,
              img_width = 256,
              n_channels = 3,
              n_classes = len(class_names)
              )

AUTOTUNE = tf.data.AUTOTUNE

val_size = int(image_count * CONFIG['test_split'])
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

def decode_array(img):
    array = tf.numpy_function(np.load, [img], tf.uint8)
    return array

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)

def process_path(file_path):
    label = get_label(file_path)
    array = decode_array(file_path)
    return array, label

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(CONFIG['batch_size'])
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

# for image, label in train_ds.take(1):
#     plt.imshow(image[0])
#     print(class_names[label[0].numpy()])


def get_model_vanilla():

    input_shape = (CONFIG['img_height'], CONFIG['img_width'], CONFIG['n_channels'])

    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Rescaling(1./255),

        tf.keras.layers.Conv2D(16, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(32, 2, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(.3),

        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(units=CONFIG['n_classes'], activation='softmax')
    ])
    return model


def get_model_custom():
    model = Sequential()

    input_shape = (CONFIG['img_height'], CONFIG['img_width'], CONFIG['n_channels'])

    model.add(layers.Input(shape=input_shape)),

    model.add(layers.Rescaling(1./255)),

    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.Dropout(0.1))

    model.add(layers.Dense(units=CONFIG['n_classes'], activation='softmax'))
    return model

def compile_model(model):
    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model


model = get_model_vanilla()
model.summary()

model = compile_model(model)

es = EarlyStopping(patience=20,
                   restore_best_weights=True)

model.fit(train_ds,
          validation_data=val_ds,
          callbacks = [es],
          epochs=500)
