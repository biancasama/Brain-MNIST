import tensorflow as tf
import pathlib
import numpy as np
import os

data_dir = pathlib.Path('/data/images')
print(data_dir)

image_count = len(list(data_dir.glob('*/*.npy')))
print(image_count)

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
print(list_ds)
# list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

# class_names = np.array(sorted([item.name for item in data_dir.glob('*/*.npy') if item.name != ".ipynb_checkpoints"]))
# class_names = np.array([label.replace('.npy', '').split('_')[-1] for label in class_names])

# val_size = int(image_count * 0.3)
# train_ds = list_ds.skip(val_size)
# val_ds = list_ds.take(val_size)

# AUTOTUNE = tf.data.AUTOTUNE
# img_height = 192
# img_width = 256

# def get_label(file_path):
#   # Convert the path to a list of path components
#   parts = tf.strings.split(file_path, os.path.sep)
#   # The second to last is the class-directory
#   file_name = parts[-1]
#   label = tf.strings.split(file_name, sep='_', maxsplit=-1, name=None)[-1]

#   label = tf.strings.regex_replace(
#     label, '.npy', '', replace_global=True, name=None)

#   return label

# def decode_img(img):
#   image = tf.numpy_function(np.load, [img], tf.uint8)
#   return image

# def process_path(file_path):
#   label = get_label(file_path)

#   # Load the raw data from the file as a string
#   # img = tf.io.read_file(file_path)
#   img = decode_img(file_path)

#   return img, label

# train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
# val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# for image, label in train_ds.take(1):
#   print("Image shape: ", image.numpy().shape)
#   print("Label: ", label.numpy())
