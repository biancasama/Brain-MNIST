#tensorflow
from re import A
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model, models
from tensorflow.keras.layers import Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Flatten, LSTM, Masking, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Masking, Normalization, InputLayer
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
from other_data import download_blob, upload_blob
import matplotlib.pyplot as plt
import sklearn


def prepare_for_RNN_4C_otherData():

    ##retrieve X and y saved as blobs in bucket
    BUCKET_NAME = "brain-mnist"
    download_blob(BUCKET_NAME, f'data/{dataset_name}_filtered_{detail}_X.npy', f"other_datasets/{dataset_name}_filtered_{detail}_X.npy")
    download_blob(BUCKET_NAME, f'data/{dataset_name}_filtered_{detail}_y.npy', f"other_datasets/{dataset_name}_filtered_{detail}_y.npy")
    X = np.load(f'data/{dataset_name}_filtered_{detail}_X.npy', allow_pickle=True, fix_imports=True)
    y = np.load(f'data/{dataset_name}_filtered_{detail}_y.npy', allow_pickle=True, fix_imports=True)
    print(X.shape)
    print(y.shape)

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

    model.add(normalizer)
    model.add(InputLayer(input_shape=(X_train.shape[1],14)))
    model.add(LSTM(units=50, activation='tanh', return_sequences=True))
    model.add(LSTM(units=80, activation='tanh', return_sequences=True))
    model.add(LSTM(units=50, activation='tanh'))

    model.add(layers.Dense(50, activation="relu"))
    #layers.Dropout(0.2)
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(11, activation="softmax"))
    # model.add(layers.Dense(10, activation="softmax"))
    #
    model.summary()

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
    mlflow_experiment = f'mnist_experiment_fla66_{dataset_name}_{detail2}'
    mlflow_model_name = f'mnist_fla66_{dataset_name}_{detail2}'

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




def plot_loss_accuracy(history):
    with plt.style.context('seaborn-deep'):

        fig, ax = plt.subplots(1, 2, figsize=(15, 4))

        ## Plot Losses and Accuracies
        x_axis = np.arange(len(history.history['loss']))

        ax[0].set_title("Loss")
        ax[0].plot(x_axis, history.history['loss'], color="blue", linestyle=":", marker="X", label="Train Loss")
        ax[0].plot(x_axis, history.history['val_loss'], color="orange", linestyle="-", marker="X", label="Val Loss")

        ax[1].set_title("Accuracy")
        ax[1].plot(x_axis, history.history['accuracy'], color="blue", linestyle=":", marker="X", label="Train Accuracy")
        ax[1].plot(x_axis,
                   history.history['val_accuracy'],
                   color="orange",
                   linestyle="-",
                   marker="X",
                   label="Val Accuracy")

        ## Customization
        ax[0].grid(axis="x", linewidth=0.5)
        ax[0].grid(axis="y", linewidth=0.5)
        ax[0].legend()
        ax[1].grid(axis="x", linewidth=0.5)
        ax[1].grid(axis="y", linewidth=0.5)
        ax[1].legend()

        return fig




def train_model_RNN_4C_otherData(model, X_train, y_train):

    # model params
    batch_size = 256
    patience = 20
    epochs = 500

    es = EarlyStopping(monitor="val_loss",
                       patience=patience,
                       restore_best_weights=True,
                       verbose=0)

    chkpt = ModelCheckpoint(filepath=f'checkpoints/model_checkpoint_{dataset_name}_{detail2}',
                            save_weights_only=True,
                            save_best_only=True,
                            monitor='val_loss',
                            mode='min')

    history = model.fit(X_train, y_train,
                        validation_split=0.2,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[es,chkpt],
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

    ### plot learning curves ###
    fig = plot_loss_accuracy(history)

    #save in bucket
    BUCKET_NAME = "brain-mnist"
    fig.savefig(f"results/RNN_{dataset_name}_{detail2}.png") #save png locally
    upload_blob(BUCKET_NAME, f'results/RNN_{dataset_name}_{detail2}.png', f"results/RNN_{dataset_name}_{detail2}.png")

    return val_accuracy




def evaluate_model_RNN_4C_otherData(model, X_test, y_test):

    res = model.evaluate(X_test, y_test, verbose=0)
    print(res)

    predicted_probabilities = model.predict(X_test)
    print(predicted_probabilities)

    y_pred = np.argmax(predicted_probabilities, axis = 1)
    print(y_pred)
    y_test = np.argmax(y_test, axis = 1)
    print(y_test)

    matrix_conf = sklearn.metrics.confusion_matrix(y_test, y_pred)
    print(matrix_conf)

    matrix_conf_all = sklearn.metrics.confusion_matrix(y_test, y_pred, normalize='all')
    print(matrix_conf_all)

    # sklearn.metrics.ConfusionMatrixDisplay(matrix_conf).savefig(f"results/conf_matrix_{dataset_name}_{detail}.png")

    BUCKET_NAME = "brain-mnist"
    np.save(f'results/res_{dataset_name}_{detail2}.npy', res, allow_pickle=True, fix_imports=True) #save y locally
    np.save(f'results/conf_matrix_{dataset_name}_{detail2}.npy', matrix_conf, allow_pickle=True, fix_imports=True) #save X locally
    np.save(f'results/conf_matrix_all_{dataset_name}_{detail2}.npy', matrix_conf_all, allow_pickle=True, fix_imports=True) #save y locally
    np.save(f'results/ypred_{dataset_name}_{detail2}.npy', y_pred, allow_pickle=True, fix_imports=True) #save X locally
    np.save(f'results/ytest_{dataset_name}_{detail2}.npy', y_test, allow_pickle=True, fix_imports=True) #save y locally

    upload_blob(BUCKET_NAME, f'results/res_{dataset_name}_{detail2}.npy', f"results/res_{dataset_name}_{detail2}.npy")
    upload_blob(BUCKET_NAME, f'results/conf_matrix_{dataset_name}_{detail2}.npy', f"results/conf_matrix_{dataset_name}_{detail2}.npy")
    upload_blob(BUCKET_NAME, f'results/conf_matrix_all_{dataset_name}_{detail2}.npy', f"results/conf_matrix_all_{dataset_name}_{detail2}.npy")
    upload_blob(BUCKET_NAME, f'results/ypred_{dataset_name}_{detail2}.npy', f"results/ypred_{dataset_name}_{detail2}.npy")
    upload_blob(BUCKET_NAME, f'results/ytest_{dataset_name}_{detail2}.npy', f"results/ytest_{dataset_name}_{detail2}.npy")

    return matrix_conf


def load_model_otherData() -> Model:
    """
    load a saved model, return None if no model found
    """
    # stage = "Production"

    # load model from mlflow
    mlflow_tracking_uri = 'https://mlflow.lewagon.ai'
    mlflow_model_name = f'mnist_fla66_{dataset_name}_{detail2}'

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    model_uri = f"models:/{mlflow_model_name}/4"

    model = mlflow.keras.load_model(model_uri=model_uri)

    return model


if __name__=='__main__':

    from datetime import datetime

    # Getting the current date and time
    dt = datetime.now()

    # getting the timestamp
    ts = datetime.timestamp(dt)

    dataset_name = 'EP1.01'
    detail = f'cut'
    detail2 = f'cut_{ts}'

    X_train, X_test, y_train, y_test = prepare_for_RNN_4C_otherData()
    print(X_train.shape)
    model = initialize_model_RNN_4C_otherData(X_train)
    model = compile_model_RNN_4C_otherData(model)
    train_model_RNN_4C_otherData(model, X_train, y_train)

    # model = load_model_otherData()

    evaluate_model_RNN_4C_otherData(model, X_test, y_test)
