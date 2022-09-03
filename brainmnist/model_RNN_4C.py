#tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model, models
from tensorflow.keras.layers import Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Flatten, LSTM, Masking, GRU
from tensorflow.keras.callbacks import EarlyStopping
#sklearn
from sklearn.model_selection import train_test_split
#numpy
import numpy as np
#environnement
import os
#mlflow
import mlflow
#internal functions
from data import load_clean_data_from_bucket, balance_data, map_data_array3D


def prepare_for_RNN_4C():

    df = load_clean_data_from_bucket()
    df = balance_data(df)
    X, y = map_data_array3D(df)

    y_copy = y.copy()
    y_copy[y_copy==-1]=10
    y_cat = to_categorical(y_copy)
    y_cat.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)

    return X_train, X_test, y_train, y_test


def initialize_model_RNN_4C():

    model = Sequential()

    model.add(layers.Masking(mask_value=-1, input_shape=(512,4)))

    model.add(LSTM(units=20, activation='tanh',return_sequences=True))
    model.add(LSTM(units=50, activation='tanh',return_sequences=True))
    model.add(LSTM(units=20, activation='tanh'))

    model.add(layers.Dense(50, activation="relu"))
    layers.Dropout(0.2)
    model.add(layers.Dense(11, activation="softmax"))

    return model


def compile_model_RNN_4C(model):

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics='accuracy')

    return model



def save_model_RNN_4C(model: Model = None,
                      params: dict = None,
                      metrics: dict = None) -> None:

    # retrieve mlflow env params
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")
    mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

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



def train_model_RNN_4C(model, X_train, y_train):

    # model params
    batch_size = 256
    patience = 2
    epochs = 1

    es = EarlyStopping(patience=patience, restore_best_weights=True)

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
        val_train=np.min(history.history['val_train']),
        val_accuracy=np.min(history.history['val_accuracy']),
        batch_size=batch_size,
        patience=patience,
        epochs=epochs
    )

    # save model
    save_model_RNN_4C(model=model, params=params, metrics=dict(val_accuracy=val_accuracy))

    return val_accuracy


if __name__=='__main__':
    X_train, X_test, y_train, y_test = prepare_for_RNN_4C()
    model = initialize_model_RNN_4C()
    model = compile_model_RNN_4C(model)
    train_model_RNN_4C(model, X_train, y_train)
    print(X_train.shape)

    # os.makedirs('results/model1', exist_ok=True)
    # model.summary.to_csv('results/model1/summary')