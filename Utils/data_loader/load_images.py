import tensorflow as tf


def load_x_y_images(img_path, origin_path, image_size=(320, 320)):
    images = tf.keras.utils.image_dataset_from_directory(
        img_path, image_size=image_size, label_mode=None)
    org_img = tf.keras.utils.image_dataset_from_directory(
        origin_path, image_size=image_size, label_mode=None)

    for _ in images:
        org_img = org_img.concatenate(org_img)

    return images, org_img

def load_x_images(img_path, origin_path, image_size=(320, 320)):
    images = tf.keras.utils.image_dataset_from_directory(
        img_path, image_size=image_size, label_mode=None)

    return images, images


def load_test_image(img_path, pic_shape):
    img = tf.keras.utils.load_img(img_path, target_size=pic_shape)
    img = tf.keras.utils.img_to_array(img)
    img = tf.expand_dims(img, 0)
    return img
