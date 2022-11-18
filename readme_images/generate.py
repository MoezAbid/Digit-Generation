import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1337)
import keras

KERAS_MODEL_FILEPATH = "./model/acgan_mnist_digits.h5"

# Loading model
generator = keras.models.load_model(KERAS_MODEL_FILEPATH)
latent_size = 100
def make_digit(digit=None):
    noise = np.random.uniform(-1, 1, (1, latent_size))

    sampled_label = np.array([
            digit if digit is not None else np.random.randint(0, 10, 1)
        ]).reshape(-1, 1)

    generated_image = generator.predict(
        [noise, sampled_label], verbose=0)

    return np.squeeze((generated_image * 127.5 + 127.5).astype(np.uint8)), noise

digit, noise = make_digit(digit=6)
plt.imshow(digit, cmap='gray_r', interpolation='nearest')
plt.axis('off')
plt.savefig("generated_digit.png")

plt.figure()
plt.imshow(noise.reshape(10, 10), cmap='gray_r', interpolation='nearest')
plt.savefig("noise_used.png")