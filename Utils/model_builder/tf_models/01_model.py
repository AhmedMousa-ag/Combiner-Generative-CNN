from tensorflow_builder import tf_builder
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Conv2DTranspose


class model(tf_builder):
    """This model is a simple autoencoder as our base model"""
    def __init__(slef,*args,**kwargs):
        super.__init__(*args,**kwargs)

    def _build_model(self):
        self.model = keras.models.Sequential([self.encoder(self.pic_shape), self.decoder(self.pic_shape)])
        self.model._compile()

    def _encoder(input_shape):
        encoder = keras.models.Sequential()
        

        encoder.add(Conv2D(16 , kernel_size=3, padding='same', activation='selu',input_shape=input_shape))
        encoder.add(MaxPool2D(pool_size=2))

        encoder.add(Conv2D(32 , kernel_size=3, padding='same', activation='selu'))
        encoder.add(MaxPool2D(pool_size=2))

        encoder.add(Conv2D(64 , kernel_size=3, padding='same', activation='selu'))
        encoder.add(MaxPool2D(pool_size=2))

        encoder.add(Conv2D(64*2 , kernel_size=3, padding='same', activation='selu'))
        encoder.add(MaxPool2D(pool_size=2))

        encoder.add(Conv2D(64*3 , kernel_size=3, padding='same'))#, activation='selu'))
        encoder.add(MaxPool2D(pool_size=2))

        return encoder

    def _decoder(pic_shape):
        decoder = keras.models.Sequential()

        decoder.add(Conv2DTranspose(64*2, kernel_size=3, strides=2, padding='valid', activation='selu',input_shape=[10, 10, 192]))
        decoder.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='valid', activation='selu'))
        decoder.add(Conv2DTranspose(32 , kernel_size=3, strides=2, padding='same',activation='selu'))
        decoder.add(Conv2DTranspose(12 , kernel_size=3, strides=2, padding='same',activation='selu'))
        decoder.add(Conv2DTranspose(3 , kernel_size=3, strides=2, padding='same',activation='selu'))

        return decoder