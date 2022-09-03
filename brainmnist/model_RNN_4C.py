#tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.layers import Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Flatten, LSTM, Masking, GRU
from tensorflow.keras.callbacks import EarlyStopping
#sklearn
from sklearn.model_selection import train_test_split
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

    model.add(LSTM(units=50, activation='tanh',return_sequences=True))
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


def train_model_RNN_4C(model, X_train, y_train):

    es = EarlyStopping(patience=2, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_split=0.2,
                        batch_size=64,
                        epochs=500,
                        callbacks=[es],
                        shuffle=True,
                        verbose=1)

    return history, model


if __name__=='__main__':
    X_train, X_test, y_train, y_test = prepare_for_RNN_4C()
    model = initialize_model_RNN_4C()
    model = compile_model_RNN_4C(model)
    history, model = train_model_RNN_4C(model, X_train, y_train)
    print(model.summary)
    print(len(history))
