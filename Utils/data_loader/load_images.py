from tkinter import image_types
import tensorflow as tf
from Utils.config import load_config_file
import os
import glob
import pandas as pd
config_file = load_config_file()
data_path = config_file["paths"]["data"]

pic_shape = config_file["train_config"]["pic_shape"]

def augment_images_brigh(x, y):
    x = tf.image.adjust_brightness(x, 0.1)
    y = tf.image.adjust_brightness(y, 0.1)
    return x, y

def augment_images_contrast(x, y):
    x = tf.image.adjust_contrast(x, 0.1)
    y = tf.image.adjust_contrast(y, 0.1)
    return x, y

def augment_images_flip(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x, y

def produce_x_y_paths(data_version):
    x_data = []
    y_data = []
    for folder in os.listdir(os.path.join(data_path, data_version))[:-1]: # -1 because we want to exclude test folder
        path = os.path.join(data_path, data_version, folder)
        origin_path = os.path.join(path, "origin")
        origin_paths = []
        origin_types = (origin_path+"/*.jpeg",origin_path+"/*.jpg")

        for type in origin_types:
            for pat in glob.glob(type):
                origin_paths.append(pat)
        origin_path = origin_paths[0] #It's one image so no problem

        image_types = (path+"/*.jpeg",path+"/*.jpg")
        images_paths = []
        for type in image_types:
            for pat in glob.glob(type):
                images_paths.append(pat)
        
        for image_path in images_paths:
            x_data.append(image_path)
            y_data.append(origin_path)

    x_data = tf.data.Dataset.from_tensor_slices(x_data)
    y_data = tf.data.Dataset.from_tensor_slices(y_data)
    dataset = tf.data.Dataset.zip((x_data,y_data))
    return dataset

def load_x_y_images(data_version, add_augmentation=False,scale=False):
    dataset = produce_x_y_paths(data_version=data_version)
    dataset = dataset.map(load_images)
    print(dataset)
    if add_augmentation:
        dataset = dataset.map(augment_images_brigh)
        #TODO Tensorflow has no proper function to append (combine two datasets together) # Concatenate doesn't work 
        #augmented_set = augmented_set.concatenate(augmented_set.map(augment_images_contrast)) 
        #augmented_set = augmented_set.concatenate(augmented_set.map(augment_images_flip))
        #dataset = dataset.concatenate(augmented_set)
    if scale:
        dataset = dataset.map(scale_pic)
    return dataset
             

def scale_pic(x_img, y_img):
    x_img = x_img / 255.
    y_img = y_img / 255.
    return x_img, y_img

def load_images(x_path,y_path):
    x_img = tf.io.read_file(x_path)
    x_img = tf.io.decode_jpeg(x_img, channels=3)
    x_img = tf.image.resize(x_img, pic_shape[:-1])

    y_img = tf.io.read_file(y_path)
    y_img = tf.io.decode_jpeg(y_img, channels=3)
    y_img = tf.image.resize(y_img, pic_shape[:-1])

    return x_img,y_img

def load_x_images(data_version, image_size=pic_shape[:-1], add_augmentation=True,scale=False):
    images = None

    for folder in os.listdir(os.path.join(data_path, data_version))[:-1]:
        path = os.path.join(data_path, data_version, folder)

        if not images:
            images = tf.keras.utils.image_dataset_from_directory(
                path, image_size=image_size, label_mode=None)
        else:
            images = images.concatenate(tf.keras.utils.image_dataset_from_directory(
                path, image_size=image_size, label_mode=None))

    dataset = tf.data.Dataset.zip((images, images))

    if add_augmentation:
        augmented_set = dataset.map(augment_images)
        dataset = dataset.concatenate(augmented_set)
    if scale:
        dataset = dataset.map(scale_pic)
    return dataset 


def load_test_image(img_path, pic_shape):
    img = tf.keras.utils.load_img(img_path, target_size=pic_shape)
    img = tf.keras.utils.img_to_array(img)
    img = tf.expand_dims(img, 0)
    return img
