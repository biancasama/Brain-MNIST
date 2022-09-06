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
# gsutil -m cp gs://brain-mnist/data/nothing/\*.npy data/images/nothing

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

data_dir = pathlib.Path(f'{os.getenv("HOME")}/code/fla66/Brain-MNIST/data/images')
print(data_dir)

image_count = len(list(data_dir.glob('*/*.npy')))
print(image_count)

# list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
# list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

# class_names = np.array(sorted([item.name for item in data_dir.glob('*/*.npy') if item.name != ".ipynb_checkpoints"]))
# class_names = np.array([label.replace('.npy', '').split('_')[-1] for label in class_names])
# class_names = [class_.encode('utf-8') for class_ in class_names]
# class_names = np.unique(class_names)

# val_size = int(image_count * 0.3)
# train_ds = list_ds.skip(val_size)
# val_ds = list_ds.take(val_size)

# AUTOTUNE = tf.data.AUTOTUNE
# img_height = 192
# img_width = 256


# def decode_img(img):
#     image = tf.numpy_function(np.load, [img], tf.uint8)
#     return tf.expand_dims(image, 0)


# def get_label(file_path):
#     # Convert the path to a list of path components
#     parts = tf.strings.split(file_path, os.path.sep)
#     # The second to last is the class-directory
#     file_name = parts[-1]
#     label = tf.strings.split(file_name, sep='_', maxsplit=-1, name=None)[-1]

#     label = tf.strings.regex_replace(
#     label, '.npy', '', replace_global=True, name=None)

#     one_hot = label == class_names
#     # Integer encode the label
#     return tf.expand_dims(tf.reshape(one_hot, (1,4)), 0)



# def process_path(file_path):
#     label = get_label(file_path)

#     # Load the raw data from the file as a string
#     # img = tf.io.read_file(file_path)
#     img = decode_img(file_path)

#     return img, label


# train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
# val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)


# for image, label in train_ds.take(1):
#     print("Image shape: ", image.numpy().shape)
#     print("Label: ", label)


# from tensorflow.keras import Sequential, layers
# from tensorflow.keras.layers import Dense, Reshape

# model = Sequential()

# input_shape = (192, 256, 3)

# model.add(layers.Conv2D(32, (5, 5), input_shape=input_shape, activation='relu'))
# model.add(layers.Conv2D(32, (5, 5), activation='relu'))
# model.add(layers.MaxPool2D(pool_size=(2, 2)))
# model.add(layers.Dropout(0.1))

# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPool2D(pool_size=(2, 2)))
# model.add(layers.Dropout(0.1))

# model.add(layers.Flatten())
# model.add(Dense(units=128, activation='relu'))
# model.add(layers.Dropout(0.1))

# model.add(Dense(units=64, activation='relu'))
# model.add(layers.Dropout(0.1))

# model.add(Dense(units=4, activation='softmax'))
# model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

# model.fit(train_ds, validation_data=val_ds, epochs=3, batch_size=2)
