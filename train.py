from PIL import Image
import os
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import Multiply, LeakyReLU, UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import wandb
np.random.seed(1337)
wandb.init(project="ACGAN-Digit-Generation")
os.system("mkdir generated_digits_plots")
os.system("mkdir model")
KERAS_MODEL_FILEPATH = "./model/acgan_mnist_digits.h5"


# Generator
def build_generator(latent_size):
    cnn = Sequential()
    cnn.add(Dense(1024, input_dim=latent_size, activation='relu'))
    cnn.add(Dense(128 * 7 * 7, activation='relu'))
    cnn.add(Reshape((7, 7, 128)))
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(256, 5, padding='same', activation='relu',
                   kernel_initializer='glorot_normal'))
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(128, 5, padding='same', activation='relu',
                   kernel_initializer='glorot_normal'))
    cnn.add(Conv2D(1, 2, padding='same', activation='tanh',
                   kernel_initializer='glorot_normal'))
    latent = Input(shape=(latent_size,))
    image_class = Input(shape=(1,), dtype='int32')
    # 10 classes in MNIST
    emb = Embedding(10, latent_size, embeddings_initializer='glorot_normal')(image_class)
    cls = Flatten()(emb)
    h = Multiply()([latent, cls])
    fake_image = cnn(h)
    return Model([latent, image_class], fake_image)

def build_discriminator():
    cnn = Sequential()
    cnn.add(Conv2D(32, 3, padding='same', strides=2, input_shape=(28, 28, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))
    cnn.add(Conv2D(64, 3, padding='same', strides=1))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))
    cnn.add(Conv2D(128, 3, padding='same', strides=2))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))
    cnn.add(Conv2D(256, 3, padding='same', strides=1))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))
    cnn.add(Flatten())
    image = Input(shape=(28, 28, 1))
    features = cnn(image)
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(10, activation='softmax', name='auxiliary')(features)
    return Model(image, [fake, aux])

# Building
epochs = 50
batch_size = 64
latent_size = 100
adam_lr = 0.00005
adam_beta_1 = 0.5

config = dict(
        epochs = 50,
        batch_size = 100,
        latent_size = 100,
        adam_lr = 0.00005,
        adam_beta_1 = 0.5,
)
wandb.config = config

# build the discriminator
discriminator = build_discriminator()
discriminator.compile(
    optimizer=Adam(learning_rate=adam_lr, beta_1=adam_beta_1),
    loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
)

# build the generator
generator = build_generator(latent_size)
generator.compile(
    optimizer=Adam(learning_rate=adam_lr, beta_1=adam_beta_1),
    loss='binary_crossentropy'
)

generator.summary()
discriminator.summary()

import tensorflow as tf
def draw_model_architecture(model, file_name = "model.png"):
    model_img_file = file_name
    return tf.keras.utils.plot_model(model, to_file=model_img_file, 
                            show_shapes=True, 
                            show_layer_activations=True, 
                            show_dtype=True,
                            show_layer_names=True)

# draw_model_architecture(generator, "generator.png")
# draw_model_architecture(discriminator, "discriminator.png")

# Data
# mnist data, and force it to be of shape (..., 28, 28, 1) with
# range [-1, 1]
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
X_test = np.expand_dims(X_test, axis=3)
num_train, num_test = X_train.shape[0], X_test.shape[0]

# ACGAN
latent = Input(shape=(latent_size, ))
image_class = Input(shape=(1,), dtype='int32')

# get a fake image
fake = generator([latent, image_class])

# we only want to be able to train generation for the combined model
discriminator.trainable = False
fake, aux = discriminator(fake)
combined = Model([latent, image_class], [fake, aux])

combined.compile(
    optimizer=Adam(learning_rate=adam_lr, beta_1=adam_beta_1),
    loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
)

def get_generator():
    return generator

# draw_model_architecture(combined, "ACGAN.png")

# Training
print('Epoch\tL_s(G)\tL_s(G)\tL_s(D)\tL_s(D)\tL_c(G)\tL_c(G)\tL_c(D)\tL_c(D)')
from tqdm import tqdm
for epoch in tqdm(range(epochs)):
    print(epoch + 1, end='\t', flush=True)

    num_batches = int(X_train.shape[0] / batch_size)

    epoch_gen_loss = []
    epoch_disc_loss = []

    for index in range(num_batches):
        # generate a new batch of noise
        noise = np.random.uniform(-1, 1, (batch_size, latent_size))

        # get a batch of real images
        image_batch = X_train[index * batch_size:(index + 1) * batch_size]
        label_batch = y_train[index * batch_size:(index + 1) * batch_size]

        # sample some labels from p_c
        sampled_labels = np.random.randint(0, 10, batch_size)

        # generate a batch of fake images, using the generated labels as a
        # conditioner. We reshape the sampled labels to be
        # (batch_size, 1) so that we can feed them into the embedding
        # layer as a length one sequence
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=0)

        X = np.concatenate((image_batch, generated_images))
        y = np.array([1] * batch_size + [0] * batch_size)
        aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

        # see if the discriminator can figure itself out...
        epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))

        # make new noise. we generate 2 * batch size here such that we have
        # the generator optimize over an identical number of images as the
        # discriminator
        noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
        sampled_labels = np.random.randint(0, 10, 2 * batch_size)

        # we want to train the generator to trick the discriminator
        # For the generator, we want all the {fake, not-fake} labels to say
        # not-fake
        trick = np.ones(2 * batch_size)

        epoch_gen_loss.append(
            combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))],
                [trick, sampled_labels]
            )
        )

    # evaluate the testing loss here

    # generate a new batch of noise
    noise = np.random.uniform(-1, 1, (num_test, latent_size))

    # sample some labels from p_c and generate images from them
    sampled_labels = np.random.randint(0, 10, num_test)
    generated_images = generator.predict(
        [noise, sampled_labels.reshape((-1, 1))], verbose=0)

    X = np.concatenate((X_test, generated_images))
    y = np.array([1] * num_test + [0] * num_test)
    aux_y = np.concatenate((y_test, sampled_labels), axis=0)

    # see if the discriminator can figure itself out...
    discriminator_test_loss = discriminator.evaluate(X, [y, aux_y], verbose=0)

    discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

    # make new noise
    noise = np.random.uniform(-1, 1, (2 * num_test, latent_size))
    sampled_labels = np.random.randint(0, 10, 2 * num_test)

    trick = np.ones(2 * num_test)

    generator_test_loss = combined.evaluate(
        [noise, sampled_labels.reshape((-1, 1))],
        [trick, sampled_labels],
        verbose=0
    )

    generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
    
    print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(
        # generation loss
        generator_train_loss[1], generator_test_loss[1],
        discriminator_train_loss[1], discriminator_test_loss[1],
        # auxillary loss
        generator_train_loss[2], generator_test_loss[2],
        discriminator_train_loss[2], discriminator_test_loss[2],
    ))

    wandb.log({"generator_train_loss":generator_train_loss[1]})
    wandb.log({"generator_test_loss":generator_test_loss[1]})
    wandb.log({"discriminator_train_loss":discriminator_train_loss[2]})
    wandb.log({"discriminator_test_loss":discriminator_test_loss[2]})

    # save model every epoch
    generator.save(KERAS_MODEL_FILEPATH)

    # generate some digits to display
    noise = np.random.uniform(-1, 1, (100, latent_size))

    sampled_labels = np.array([[i] * 10 for i in range(10)]).reshape(-1, 1)

    # get a batch to display
    generated_images = generator.predict([noise, sampled_labels], verbose=0)

    # arrange them into a grid
    img = (np.concatenate([r.reshape(-1, 28)
                           for r in np.split(generated_images, 10)
                           ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

    Image.fromarray(img).save('./generated_digits_plots/mnist_acgan_generated_{0:03d}.png'.format(epoch))
    
print('Training is finished.')