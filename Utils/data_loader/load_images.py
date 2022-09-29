import tensorflow as tf


def augment_images(x_image, y_image):
    x = x_image
    y = y_image
    x = tf.image.adjust_brightness(x, 0.1)
    y = tf.image.adjust_brightness(y, 0.1)
    x = tf.image.adjust_contrast(x, 0.1)
    y = tf.image.adjust_contrast(y, 0.1)

    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x, y


def scale_pic(x_img, y_img):
    x_img = x_img / 255.
    y_img = y_img / 255.


def load_x_y_images(img_path, origin_path, image_size=(320, 320), add_augmentation=True):
    images = tf.keras.utils.image_dataset_from_directory(
        img_path, image_size=image_size, label_mode=None)
    org_img = tf.keras.utils.image_dataset_from_directory(
        origin_path, image_size=image_size, label_mode=None)

    for _ in images:
        org_img = org_img.concatenate(org_img)

    dataset = tf.data.Dataset.zip((images, org_img))

    if add_augmentation:
        augmented_set = dataset.map(augment_images)
        dataset = dataset.concatenate(augmented_set)

    return dataset


def load_x_images(img_path, image_size=(320, 320), add_augmentation=True):
    images = tf.keras.utils.image_dataset_from_directory(
        img_path, image_size=image_size, label_mode=None)

    dataset = tf.data.Dataset.zip((images, images))

    if add_augmentation:
        augmented_set = dataset.map(augment_images)
        dataset = dataset.concatenate(augmented_set)

    return dataset


def load_test_image(img_path, pic_shape):
    img = tf.keras.utils.load_img(img_path, target_size=pic_shape)
    img = tf.keras.utils.img_to_array(img)
    img = tf.expand_dims(img, 0)
    return img
