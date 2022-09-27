import numpy as np
import matplotlib.pyplot as plt


def display_one_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image.numpy().astype("uint8"))
    plt.title("predicted image")
    plt.axis("off")
