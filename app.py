import gradio as gr
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

    # Return digit and noise for plotting
    generated_image = np.squeeze((generated_image * 127.5 + 127.5).astype(np.uint8))
    plt.figure(figsize=(3, 3))
    plt.imshow(generated_image, cmap='gray_r', interpolation='nearest')
    plt.axis('off')
    plt.savefig("generated_digit.png")

    plt.figure(figsize=(3, 3))
    plt.imshow(noise.reshape(10, 10), cmap='gray_r', interpolation='nearest')
    plt.savefig("noise_used.png")
    return "noise_used.png", "generated_digit.png"

gr.Interface(fn=make_digit,
    title="ACGAN Digit Generation",
    css="footer {visibility: hidden}",
    inputs = gr.Dropdown(choices=list(range(10)), label="Digit to generate", show_label=True),
    outputs = [ 
    gr.Image(shape=(3,3), label="Noise in use").style(full_width=True, height=250, width=725), 
    gr.Image(shape=(3,3), label="Generated digit").style(full_width=True, height=250, width=725)
    ]
).launch()   