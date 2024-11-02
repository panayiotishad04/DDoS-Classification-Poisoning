import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras as k
from tensorflow.keras import layers as l, models as m
import shap

from utils import COLUMNS, get_train_dataset, get_test_dataset, create_csv

np.random.seed(42)
tf.random.set_seed(42)


def get_trained_model():
    model = make_classifier_model()

    model.compile(optimizer='adam',
                  loss="binary_crossentropy",
                  metrics=['accuracy'])

    model.summary()

    x_train, y_train = get_train_dataset()
    model.fit(x_train, y_train, epochs=50, batch_size=32, shuffle=True, verbose=0)

    return model


def make_classifier_model():
    model = m.Sequential([
        l.Input(shape=(len(COLUMNS),)),
        l.Dense(64, activation='relu'),
        l.Dense(32, activation='relu'),
        l.Dense(1, activation='sigmoid')
    ])

    return model


def make_generator_model():
    model = m.Sequential([
        l.Input(shape=(100,)),
        l.Dense(32, activation='relu'),
        l.Dense(64, activation='relu'),
        l.Dense(len(COLUMNS), activation='tanh')
    ])

    return model


if __name__ == "__main__":

    def explain_classifier_model():
        x, y = get_train_dataset()
        x_test, y = get_test_dataset()

        model = get_trained_model()

        subset = x_test[:10]

        explainer = shap.Explainer(model, x[:100])
        shap_values = explainer(subset)
        print(shap_values)

        for shap_value in shap_values:
            shap.plots.waterfall(shap_value)


    def make_prediction():
        x_test, y_test = get_test_dataset()
        model = get_trained_model()
        print(model.evaluate(x_test, y_test, return_dict=True))


    # Download the data and place in the same folder as "data.zip" and uncomment the following line
    # create_csv()
    explain_classifier_model()

    # make_prediction()
