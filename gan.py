import tensorflow as tf

from model import make_generator_model, NOISE_DIM, get_trained_model

BATCH_SIZE = 256

cross_entropy = tf.keras.losses.BinaryCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 500

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


def get_trained_gan():
    for epoch in range(EPOCHS):
        train_step()

        seed = tf.random.normal([1, NOISE_DIM])
        flow = generator(seed, training=False)
        predicted = discriminator.predict(flow)
        print(f"Flow {flow}, {predicted}")

    return generator, discriminator
