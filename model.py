from tensorflow.keras import layers as l, models as m

from utils import get_train_dataset, COLUMNS

NOISE_DIM = 100


def get_trained_model():
    model = make_classifier_model()

    model.compile(optimizer='adam',
                  loss="binary_crossentropy",
                  metrics=['accuracy'])

    model.summary()

    x_train, y_train = get_train_dataset()
    model.fit(x_train, y_train, epochs=100, batch_size=32, shuffle=True)

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
        l.Input(shape=(NOISE_DIM,)),
        l.Dense(64, activation='relu'),
        l.Dense(32, activation='relu'),
        l.Dense(len(COLUMNS), activation='tanh')
    ])

    return model
