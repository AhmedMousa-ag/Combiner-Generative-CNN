import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def display_one_image(image,fig_size=(5,5)):
    img = tf.squeeze(image)
    plt.figure(figsize=fig_size)
    plt.imshow(img)
    plt.title("predicted image")
    plt.axis("off")
