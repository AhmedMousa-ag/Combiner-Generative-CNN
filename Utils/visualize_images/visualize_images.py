import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def display_one_image(image):
    img = tf.squeeze(image)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title("predicted image")
    plt.axis("off")
