import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as l, models as m
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
pd.set_option('future.no_silent_downcasting', True)

TRAIN_COUNT = 1500
TEST_COUNT = 400

NOISE_DIM = 100
BATCH_SIZE = 32
EPOCHS = 20

# history_string embedding
MAX_LEN = 15
VOCAB_SIZE = 15
embedding_dim = 20

CATEGORICAL = [
    'proto_enum',
    'service_string',
    'conn_state_string',
]

COLUMNS = CATEGORICAL + [
    # 'id.orig_port',
    'id.resp_pport',
    'orig_bytes_count',
    'resp_bytes_count',
    'missed_bytes_count',
    'orig_pkts_count',
    'orig_ip_bytes_count',
    'resp_pkts_count',
    'resp_bytes'
]

CATEGORICAL_VALUES = {}


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
        return df, variables_map

    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        df.replace("-", 0, inplace=True)
        df['duration_interval'] = df['duration_interval'].astype(float)
        global CATEGORICAL_VALUES
        for column in CATEGORICAL:
            df[column], values = categorical_variable(df[column])
            CATEGORICAL_VALUES[column] = values
        return df

    def create_small(name, n):
        good = ["Only_Benign_7-1.csv", "Only_Benign_34-1.csv", "Only_Benign_60-1.csv"]
        bad = ["Only_DDOS_7-1.csv", "Only_DDOS_34-1.csv"]

        df = [pd.read_csv(file) for file in good] + [pd.read_csv(file) for file in bad]
        df = pd.concat(df, ignore_index=True)

        print(df.columns)
        print(df.dtypes)
        print(f"good count {df.shape[0]}")

        df = preprocess(df)
        small_good = df[df['Category'] == "Benign"].sample(n=n, random_state=RANDOM_SEED)
        small_bad = df[df['Category'] == "Malicious"].sample(n=n, random_state=RANDOM_SEED)

        pd.DataFrame(small_bad).to_csv(f"bad_{name}.txt")
        pd.DataFrame(small_good).to_csv(f"good_{name}.txt")

    create_small("train", TRAIN_COUNT)
    create_small("test", TEST_COUNT)


def tokenize(df):
    print(df)
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(df)
    sequences = tokenizer.texts_to_sequences(df)
    sequences = pad_sequences(sequences, maxlen=MAX_LEN)
    print(sequences.shape)
    return sequences


def get_dataset(dataset_type: str, embedding=False):
    """
    Loads dataset, create the datasets before using create_csv
    """
    good = pd.read_csv(f"good_{dataset_type}.txt")
    bad = pd.read_csv(f"bad_{dataset_type}.txt")

    df = pd.concat([good, bad])

    df_num = df[COLUMNS].values
    df_emb = tokenize(df['history_string'])

    y = df['Category'] == "Malicious"

    print("Good\n", good[COLUMNS].describe())
    print("\n\n")
    print("Bad\n", bad[COLUMNS].describe())

    if embedding:
        return [df_num, df_emb], y
    else:
        return df_num, y


def make_classifier_model():
    numerical = l.Input(shape=(len(COLUMNS),))

    x = l.Dense(128, activation='relu')(numerical)
    x = l.Dense(64, activation='relu')(x)

    output = l.Dense(1, activation='sigmoid')(x)

    model = m.Model(inputs=numerical, outputs=output)

    return model


def make_classifier_with_embedding_model():
    history_string = l.Input(shape=(MAX_LEN,))
    numerical = l.Input(shape=(len(COLUMNS),))

    x = l.Embedding(input_dim=VOCAB_SIZE, output_dim=embedding_dim, input_length=MAX_LEN)(history_string)
    x_emb = l.Flatten()(x)

    x = l.Concatenate()([numerical, x_emb])

    x = l.Dense(128, activation='relu')(x)
    x = l.Dense(64, activation='relu')(x)

    output = l.Dense(1, activation='sigmoid')(x)

    model = m.Model(inputs=[numerical, history_string], outputs=output)

    return model


def make_generator_model():
    model = m.Sequential([
        l.Input(shape=(NOISE_DIM,)),
        l.Dense(64, activation='relu'),
        l.Dense(32, activation='relu'),
        l.Dense(len(COLUMNS), activation='tanh')
    ])

    return model


def get_trained_model(embedding=False):
    """
    generate trained classifier
    """
    model = make_classifier_with_embedding_model() if embedding else make_classifier_model()

    model.compile(optimizer='adam',
                  loss="binary_crossentropy",
                  metrics=['accuracy'])

    model.summary()

    x_train, y_train = get_dataset('train', embedding=embedding)
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)

    return model


def get_trained_gan():
    """
    generate trained gan model with the discriminator
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    generator = make_generator_model()
    discriminator = get_trained_model(embedding=False)

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
    x, y = get_dataset('train')
    x_test, y = get_dataset('test')

    model = get_trained_model()

    subset = x_test[:10]

    explainer = shap.Explainer(model, x[:100])
    shap_values = explainer(subset)
    print(shap_values)

    for shap_value in shap_values:
        shap.plots.waterfall(shap_value)


def make_prediction(embedding: bool = False):
    """
    train the model and test the performance
    """

    x_test, y_test = get_dataset('test', embedding=embedding)
    model = get_trained_model(embedding=embedding)

    print(model.evaluate(x_test, y_test, return_dict=True))


def categorical_sampler(value, pop_size):
    return np.random.randint(low=0, high=len(CATEGORICAL_VALUES[value]),
                             size=pop_size)


def numerical_sampler(value, pop_size):
    return np.random.randint(low=0, high=5000, size=pop_size)


def categorical_mutator(value, name):
    # if value >= len(CATEGORICAL_VALUES[value]):
    #     return value - 1
    #
    # mutation = np.random.randint(low=-1, high=1)
    # return np.clip(mutation + value, a_min=0, a_max=len(CATEGORICAL_VALUES[name]))
    return np.random.randint(low=0, high=len(CATEGORICAL_VALUES[name]), size=1)


def numerical_mutator(value, name):
    # mutation = np.random.randint(low=-100, high=100)
    # return np.clip(mutation + value, a_min=0, a_max=2 ^ 16 - 1)
    return np.random.randint(low=0, high=777, size=1)


GENERATORS = {
    'proto_enum': categorical_sampler,
    'service_string': categorical_sampler,
    'conn_state_string': categorical_sampler,
    'id.orig_port': numerical_sampler,
    'id.resp_pport': numerical_sampler,
    'orig_bytes_count': numerical_sampler,
    'resp_bytes_count': numerical_sampler,
    'missed_bytes_count': numerical_sampler,
    'orig_pkts_count': numerical_sampler,
    'orig_ip_bytes_count': numerical_sampler,
    'resp_pkts_count': numerical_sampler,
    'resp_bytes': numerical_sampler
}

MUTATORS = {
    'proto_enum': categorical_mutator,
    'service_string': categorical_mutator,
    'conn_state_string': categorical_mutator,
    'id.orig_port': numerical_mutator,
    'id.resp_pport': numerical_mutator,
    'orig_bytes_count': numerical_mutator,
    'resp_bytes_count': numerical_mutator,
    'missed_bytes_count': numerical_mutator,
    'orig_pkts_count': numerical_mutator,
    'orig_ip_bytes_count': numerical_mutator,
    'resp_pkts_count': numerical_mutator,
    'resp_bytes': numerical_mutator
}


def genetic_algorithm():
    def generate_population(pop_size, sample_shape):
        sample = np.zeros((pop_size, *sample_shape))
        for idx, value in enumerate(COLUMNS):
            sample[:, idx] = GENERATORS[value](value, pop_size)
        return sample

    def fitness_function(model, samples, target_label):
        print(samples.shape)
        predictions = model.predict(samples)
        return np.abs(predictions - target_label)

    def select_mating_pool(samples, fitness, num_parents):
        parents = np.zeros((num_parents, samples.shape[1]))
        for parent_idx in range(num_parents):
            min_fitness_idx = np.argmin(fitness)
            parents[parent_idx, :] = samples[min_fitness_idx, :]
            fitness[min_fitness_idx] = float('inf')
        return parents

    def crossover(parents, offspring_size):
        offspring = np.zeros(offspring_size)
        crossover_point = np.uint8(offspring_size[1] / 2)

        for k in range(offspring_size[0]):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def mutation(offspring, mutation_rate=0.4):
        for idx in range(offspring.shape[0]):
            columns = random.sample(list(enumerate(COLUMNS)), int(len(COLUMNS) * mutation_rate))
            sample = offspring[idx]

            if idx % 1000 == 0:
                print(sample)
                print(columns)

            for j, value in columns:
                sample[j] = MUTATORS[value](sample, value)

            offspring[idx] = sample
            if idx % 1000 == 0:
                print(sample)

        return offspring

    # Model, target label (1 for adversarial), and parameters
    target_label = 1
    pop_size = 100
    num_generations = 1000
    num_parents_mating = 50

    # Generate initial population
    sample_shape = (len(COLUMNS),)
    samples = generate_population(pop_size, sample_shape)
    model = get_trained_model(embedding=False)

    for generation in range(num_generations):
        fitness = fitness_function(model, samples, target_label)
        parents = select_mating_pool(samples, fitness, num_parents_mating)
        offspring_crossover = crossover(parents, (pop_size - parents.shape[0], sample_shape[0]))
        offspring_mutation = mutation(offspring_crossover)
        samples[:parents.shape[0], :] = parents
        samples[parents.shape[0]:, :] = offspring_mutation

        if generation % 100 == 0:
            print(f'Generation {generation}: Best fitness = {np.min(fitness)}')

    print(samples)


if __name__ == "__main__":
    create_csv()
    genetic_algorithm()
    # make_prediction(embedding=True)
    print(COLUMNS)
    # generator, predictor = get_trained_gan()
    #
    # flow = generator(tf.random.normal([1, NOISE_DIM]), training=False)
    #
    # predicted = predictor.predict(flow)
    # print(f"Flow {flow}, prediction {predicted}")
