#tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model, models
from tensorflow.keras.layers import Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Flatten, LSTM, Masking, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Masking, Normalization
#sklearn
from sklearn.model_selection import train_test_split
#numpy
import numpy as np
#pandas
import pandas as pd
#environnement
import os
#mlflow
import mlflow
#gcp
from google.cloud import storage
#internal functions
from data import load_clean_data_from_bucket, balance_data, map_data_array3D
from other_data import download_blob


def prepare_for_RNN_4C_otherData():

    ##retrieve X and y saved as blobs in bucket
    BUCKET_NAME = "brain-mnist"
    download_blob(BUCKET_NAME, f'data/{dataset_name}_filtered_X.npy', f"other_datasets/{dataset_name}_filtered_X.npy")
    download_blob(BUCKET_NAME, f'data/{dataset_name}_filtered_y.npy', f"other_datasets/{dataset_name}_filtered_y.npy")
    X = np.load(f'data/{dataset_name}_filtered_X.npy', allow_pickle=True, fix_imports=True)
    y = np.load(f'data/{dataset_name}_filtered_y.npy', allow_pickle=True, fix_imports=True)

    #pad data
    X_pad = pad_sequences(X, dtype='float32', padding='post', value=-1000)  # int32 by default, default value=0

    #encode y
    y_copy = y.copy()
    y_copy[y_copy==-1]=10
    y_cat = to_categorical(y_copy)
    y_cat.shape

    X_train, X_test, y_train, y_test = train_test_split(X_pad, y_cat, test_size=0.2)

    return X_train, X_test, y_train, y_test



def initialize_model_RNN_4C_otherData(X_train):

    normalizer = Normalization() # Instantiate a "normalizer" layer
    normalizer.adapt(X_train) # "Fit" it on the train set

    model = Sequential()

    model.add(LSTM(units=50, activation='tanh', return_sequences=True, input_shape=(512,4)))
    model.add(LSTM(units=80, activation='tanh', return_sequences=True))
    model.add(LSTM(units=50, activation='tanh'))

    model.add(layers.Dense(50, activation="relu"))
    layers.Dropout(0.3)
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(11, activation="softmax"))

    return model



def compile_model_RNN_4C_otherData(model):

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics='accuracy')

    return model



def save_model_RNN_4C_otherData(model: Model = None,
                                params: dict = None,
                                metrics: dict = None) -> None:

    # retrieve mlflow env params
    # mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    # mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")
    # mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

    mlflow_tracking_uri = 'https://mlflow.lewagon.ai'
    mlflow_experiment = f'mnist_experiment_fla66_{dataset_name}'
    mlflow_model_name = f'mnist_fla66_{dataset_name}'

    # configure mlflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name=mlflow_experiment)

    with mlflow.start_run():

        # STEP 1: push parameters to mlflow
        mlflow.log_params(params)

        # STEP 2: push metrics to mlflow
        mlflow.log_metrics(metrics)

        # STEP 3: push model to mlflow
        if model is not None:
            mlflow.keras.log_model(keras_model=model,
                                    artifact_path="model",
                                    keras_module="tensorflow.keras",
                                    registered_model_name=mlflow_model_name)

    return None



def train_model_RNN_4C_otherData(model, X_train, y_train):

    # model params
    batch_size = 256
    patience = 5
    epochs = 200

    es = EarlyStopping(monitor="val_loss",
                       patience=patience,
                       restore_best_weights=True,
                       verbose=0)

    history = model.fit(X_train, y_train,
                        validation_split=0.2,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[es],
                        shuffle=True,
                        verbose=1)

    # return the last value of the validation accuracy
    val_accuracy = history.history['val_accuracy'][-1]

    params = dict(
        # model parameters
        train_accuracy=np.min(history.history['accuracy']),
        val_accuracy=np.min(history.history['val_accuracy']),
        batch_size=batch_size,
        patience=patience,
        epochs=epochs
    )

    # save model
    save_model_RNN_4C_otherData(model=model, params=params, metrics=dict(val_accuracy=val_accuracy))

    return val_accuracy



def load_model_otherData() -> Model:
    """
    load a saved model, return None if no model found
    """
    # stage = "Production"

    # load model from mlflow
    mlflow_tracking_uri = 'https://mlflow.lewagon.ai'
    mlflow_model_name = 'mnist_fla66_EPOC'

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    model_uri = f"models:/{mlflow_model_name}/1"

    model = mlflow.keras.load_model(model_uri=model_uri)

    return model


if __name__=='__main__':

    dataset_name = 'EP1.01'

    X_train, X_test, y_train, y_test = prepare_for_RNN_4C_otherData()
    print(X_train.shape)
    model = initialize_model_RNN_4C_otherData(X_train)
    model = compile_model_RNN_4C_otherData(model)
    train_model_RNN_4C_otherData(model, X_train, y_train)
