from tensorflow.keras import layers as l, models as m
import tensorflow as tf
import numpy as np
import pandas as pd

np.random.seed(42)
tf.random.set_seed(42)
pd.set_option('future.no_silent_downcasting', True)

TRAIN_COUNT = 500
TEST_COUNT = 1000

NOISE_DIM = 100
BATCH_SIZE = 256
EPOCHS = 300

CATEGORICAL = [
    'proto_enum',
    'service_string',
    'conn_state_string',
]

COLUMNS = CATEGORICAL + [
    # 'id.orig_port',
    # 'id.resp_pport',
    'orig_bytes_count',
    'resp_bytes_count',
    'missed_bytes_count',
    'orig_pkts_count',
    'orig_ip_bytes_count',
    'resp_pkts_count',
    'resp_bytes'
]


def create_csv():
    """
    given "Only_Benign_7-1.csv", "Only_Benign_34-1.csv", "Only_Benign_60-1.csv", "Only_DDOS_7-1.csv", "Only_DDOS_34-1.csv"
    will create bad and good data sets for training and testing.

    - the categorical values will be mapped to numbers.
    - the unknown values will be set to 0.
    """

    def categorical_variable(df: pd.DataFrame) -> pd.DataFrame:
        variables = list(df.unique())
        variables_map = dict(zip(variables, range(len(variables))))
        print(variables_map)
        df = df.apply(lambda x: variables_map[x])
        return df

    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        df.replace("-", 0, inplace=True)
        df['duration_interval'] = df['duration_interval'].astype(float)
        for column in CATEGORICAL:
            df[column] = categorical_variable(df[column])
        return df

    def create_small(name, n, random_state=42):
        good = ["Only_Benign_7-1.csv", "Only_Benign_34-1.csv", "Only_Benign_60-1.csv"]
        bad = ["Only_DDOS_7-1.csv", "Only_DDOS_34-1.csv"]

        good = [pd.read_csv(file) for file in good]
        bad = [pd.read_csv(file) for file in bad]
        good = pd.concat(good, ignore_index=True)
        bad = pd.concat(bad, ignore_index=True)

        print(good.columns)
        print(good.dtypes)
        print(f"good count {good.shape[0]}, bad count {bad.shape[0]}")

        small_good = good.sample(n=n, random_state=random_state)
        small_bad = bad.sample(n=n, random_state=random_state)

        assert good.shape[1] == bad.shape[1]

        small_good = preprocess(small_good)
        small_bad = preprocess(small_bad)

        pd.DataFrame(small_bad).to_csv(f"bad_{name}.txt")
        pd.DataFrame(small_good).to_csv(f"good_{name}.txt")

    create_small("train", TRAIN_COUNT)
    create_small("test", TEST_COUNT)


def get_train_dataset():
    """
    Load train dataset, create the datasets before using create_csv
    """
    good = pd.read_csv("good_train.txt")
    bad = pd.read_csv("bad_train.txt")

    df = pd.concat([good, bad])

    df_train = df[COLUMNS]
    y = df['Category'] == "Malicious"

    assert sum(y) == TRAIN_COUNT

    print("Good\n", good[COLUMNS].describe())
    print("\n\n")
    print("Bad\n", bad[COLUMNS].describe())

    return df_train, y


def get_test_dataset():
    """
    Load test dataset, create the datasets before using create_csv
    """
    good = pd.read_csv("good_test.txt")
    bad = pd.read_csv("bad_test.txt")

    df = pd.concat([good, bad])

    df_test = df[COLUMNS]
    y = df['Category'] == "Malicious"

    assert sum(y) == TEST_COUNT

    return df_test, y


def make_classifier_model():
    model = m.Sequential([
        l.Input(shape=(len(COLUMNS),)),
        l.Dense(128, activation='relu'),
        l.Dense(64, activation='relu'),
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


def get_trained_model():
    """
    generate trained classifier
    """
    model = make_classifier_model()

    model.compile(optimizer='adam',
                  loss="binary_crossentropy",
                  metrics=['accuracy'])

    model.summary()

    x_train, y_train = get_train_dataset()
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32, shuffle=True)

    return model


def get_trained_gan():
    """
    generate trained gan model with the discriminator
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    generator = make_generator_model()
    discriminator = get_trained_model()

    @tf.function
    def train_step():
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

        with tf.GradientTape() as gen_tape:
            generated_flows = generator(noise, training=True)
            fake_output = discriminator(generated_flows, training=False)
            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    for epoch in range(EPOCHS):
        train_step()

        seed = tf.random.normal([1, NOISE_DIM])
        flow = generator(seed, training=False)
        predicted = discriminator.predict(flow)
        print(f"Flow {flow}, {predicted}")

    return generator, discriminator


def explain_classifier_model():
    """
    use shap values to explain the detection
    """
    import shap
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
    """
    train the model and test the performance
    """
    x_test, y_test = get_test_dataset()
    model = get_trained_model()
    print(model.evaluate(x_test, y_test, return_dict=True))


if __name__ == "__main__":
    create_csv()
    make_prediction()
    print(COLUMNS)
    # generator, predictor = get_trained_gan()
    #
    # flow = generator(tf.random.normal([1, NOISE_DIM]), training=False)
    #
    # predicted = predictor.predict(flow)
    # print(f"Flow {flow}, prediction {predicted}")
