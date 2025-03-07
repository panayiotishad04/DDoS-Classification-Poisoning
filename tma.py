import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as l, models as m
from tensorflow.keras.text import Tokenizer
from tensorflow.keras.sequence import pad_sequences


np.set_printoptions(threshold=sys.maxsize)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
pd.set_option('future.no_silent_downcasting', True)

TRAIN_COUNT = 1000
TEST_COUNT = 1000

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
    'id.orig_port',
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

    def categorical_variable(df: pd.DataFrame):
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
    # Define the TextVectorization layer
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=MAX_LEN
    )

    # Adapt the layer to the text data
    vectorize_layer.adapt(df)

    # Apply vectorization
    sequences = vectorize_layer(df)

    # Convert to numpy array for inspection (optional)
    sequences = sequences.numpy()

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
    return np.random.randint(low=0, high=500, size=pop_size)


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
    return np.random.randint(low=0, high=300, size=1)


def genetic_algorithm(model, target_label):
    column_generator = {
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

    def generate_population(pop_size, sample_shape):
        sample = np.zeros((pop_size, *sample_shape))
        for idx, value in enumerate(COLUMNS):
            sample[:, idx] = column_generator[value](value, pop_size)
        return sample

    def fitness_function(model, samples, target_label):
        predictions = model.predict(samples)
        # print(predictions, samples)
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

    def mutation_algo1(offspring, mutation_rate=0.1, debug=False):
        for idx in range(offspring.shape[0]):
            for _ in range(int(mutation_rate * offspring.shape[1])):
                mutation_index = np.random.randint(0, offspring.shape[1])
                random_value = np.random.uniform(-0.5, 0.5)
                offspring[idx, mutation_index] += random_value
        return offspring

    column_mutator = {
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

    def mutation_algo2(offspring, mutation_rate=0.1, debug=False):
        for idx in range(offspring.shape[0]):
            columns = random.sample(list(enumerate(COLUMNS)), int(len(COLUMNS) * mutation_rate))
            sample = offspring[idx]

            if debug:
                print(sample)
                print(columns)

            for j, value in columns:
                sample[j] = column_mutator[value](sample, value)

            offspring[idx] = sample
            if debug:
                print(sample)

        return offspring

    pop_size = 100
    num_generations = 1000
    num_parents_mating = 50

    sample_shape = (len(COLUMNS),)
    samples = generate_population(pop_size, sample_shape)

    for generation in range(num_generations):
        fitness = fitness_function(model, samples, target_label)
        parents = select_mating_pool(samples, fitness, num_parents_mating)
        offspring_crossover = crossover(parents, (pop_size - parents.shape[0], sample_shape[0]))
        offspring_mutation = mutation_algo2(offspring_crossover, debug=False)
        samples[:parents.shape[0], :] = parents
        samples[parents.shape[0]:, :] = offspring_mutation

        if generation % 100 == 0:
            predict = model.predict(samples)
            print(samples, predict)
            print(f'Generation {generation}: Best fitness = {np.min(fitness)}')

    return samples


def boundary_algo(initial_flow, model, threshold=0.7):
    column_mutator = {
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

    def mutate_flows(flows):
        ops = []
        columns_choices = random.choices(list(enumerate(COLUMNS)), k=flows.shape[0])
        for idx, (column_idx, column_name) in zip(range(flows.shape[0]), columns_choices):
            mutator = column_mutator[column_name]
            v = flows[idx, column_idx]
            m = mutator(v, column_name)
            ops.append((m.item(), column_name))
            flows[idx, column_idx] = m
        return flows, np.array(ops).reshape((-1, 2))

    init_label = model.predict(initial_flow) > threshold
    flows, ops = mutate_flows(initial_flow.copy())
    pred_label = model.predict(flows) > threshold

    idx = pred_label != init_label

    if np.any(idx):
        init_flows = initial_flow[np.broadcast_to(idx, shape=initial_flow.shape)].reshape((-1, len(COLUMNS)))
        mut_flows = flows[np.broadcast_to(idx, shape=flows.shape)].reshape((-1, len(COLUMNS)))
        op = ops[np.broadcast_to(idx, shape=ops.shape)].reshape((-1, 2))
        pred = model.predict(init_flows)
        pred2 = model.predict(mut_flows)
        csv = np.hstack((init_flows, pred, op, pred2))
        pd.DataFrame(csv, columns=COLUMNS + ['old_pred', "value", 'column', "mutated_pred"]).to_csv('flows_§.csv')
    return initial_flow


if __name__ == "__main__":
    create_csv()


    # model = get_trained_model(embedding=False)
    # model.save('model.keras')

    def boundary_algo_run():
        model = tf.keras.models.load_model('model.keras')
        x_test, y_test = get_dataset('test', embedding=False)
        boundary_algo(x_test, model)


    boundary_algo_run()


    def genetic_algor_run():
        model = tf.keras.models.load_model('model.keras')
        print(COLUMNS)
        good = genetic_algorithm(model, 0)
        bad = genetic_algorithm(model, 1)
        good = pd.DataFrame(good, columns=COLUMNS)
        bad = pd.DataFrame(bad, columns=COLUMNS)

        pred_good = model.predict(good.values)
        pred_bad = model.predict(bad.values)
        print(good, pred_good)
        print(bad, pred_bad)
        good.to_csv('good.csv')
        bad.to_csv('bad.csv')


    def embedding_nn_run():
        make_prediction(embedding=True)


    def gan_run():
        generator, predictor = get_trained_gan()

        flow = generator(tf.random.normal([1, NOISE_DIM]), training=False)

        predicted = predictor.predict(flow)
        print(f"Flow {flow}, prediction {predicted}")
