import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def display_one_image(image,title="Predicted Image",fig_size=(5,5)):
    img = tf.squeeze(image)
    plt.figure(figsize=fig_size)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
