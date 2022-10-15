
from ast import main
from .. tensorflow_builder import tf_builder
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalizationV2, Dropout, Input, Concatenate, Lambda, ReLU
from keras.activations import tanh
from keras.models import Sequential


class __model():
    """This model is a simple autoencoder as our base model"""

    def __init__(self, *args, **kwargs):
        self.channels = 3
        self.strides = 1
        self.padding = "same"
        self.drop_rate = 0.50  # The droput rate in dropout layer
        self.de_num = 0
        self.en_num = 1

    def _build_model(self):
        self.generator = self.generator()
        self.discrimnator = self.discrimnator()

    def discrimnator(self):
        # C64-C128-C256-C512
        cnn_list = [128, 256, 512]
        model = Sequential()
        model.add(Conv2D(64, self.channels, strides=self.strides,
                         padding=self.padding))
        model.add(LeakyReLU(0.2))
        for cnn in cnn_list:
            model.add(Conv2D(cnn, self.channels, strides=self.strides,
                             padding=self.padding))
            model.add(BatchNormalizationV2())
            model.add(LeakyReLU(0.2))
        model.add(Conv2D(1, self.channels, strides=self.strides,
                         padding=self.padding, activation="sigmoid"))
        return model

    def en_block(self, input, cnn):
        self.en_num += 1
        x = Conv2D(cnn, self.channels, strides=self.strides,
                 padding=self.padding, name=f"Encoder_Conv_{cnn}_{self.en_num}")(input)
        x = BatchNormalizationV2(
            name=f"Encoder_BatchNorm_{cnn}_{self.en_num}")(x)
        x = LeakyReLU(0.2, name=f"Encoder_LRelu_{cnn}_{self.en_num}")(x)
        return x

    def de_block(self, input, cnn, dropout=True, activation=None):
        # CD512-CD512-CD512-C512-C256-C128-C64
      self.de_num += 1
      x = Conv2DTranspose(cnn, self.channels, strides=self.strides,
                          padding=self.padding, name=f"Decoder_Conv_{cnn}_{self.de_num}")(input)
      x = BatchNormalizationV2(
          name=f"Decoder_BatchNorm_{cnn}_{self.de_num}")(x)
      if activation:    # Last layer is tanh activation,
        # if we call any activation it will call this layer which is lambda layer for tanh activation
        # And doesn't have Dropout layer
        x = Lambda(lambda x: tanh(x), name=f"Decoder_tanh_{cnn}_{self.de_num}")(
            x)
      else:
        x = ReLU(0.2, name=f"Decoder_LRelu_{cnn}_{self.de_num}")(x)
        if dropout:
            x = Dropout(self.drop_rate,
                        name=f"Decoder_Dropout_{cnn}_{self.de_num}")(x)
      return x

    def encoder(self):

      en_cnn_list = [64, 128, 256, 512, 512, 512, 512, 512]  # As in paper

      input = Input(shape=[256, 256, 3], dtype=tf.float64)
      for en_C in en_cnn_list:
        if en_C == 64:
            enc = Conv2D(64, self.channels, strides=self.strides,
                       padding=self.padding, name=f"Encoder_Conv_{64}_{self.en_num}")(input)
            enc = LeakyReLU(0.2, name=f"Encoder_LRelu_{64}_{self.en_num}")(enc)
        else:
            enc = self.en_block(enc, en_C)

      model = keras.Model(inputs=input, outputs=enc)
      return model

    def generator(self):
        encoder = self.encoder()
        #CD512-CD512-CD512-C512-C256-C128-C64
        de_cnn_list = [512, 512, 512, 512, 512, 256, 128, 64]  # As in paper
        ly_range = len(de_cnn_list)
        for i, cnn in enumerate(de_cnn_list):
            if i == 0:
                de_input = self.de_block(encoder.output, cnn)
            elif i == 1:
                de = Concatenate()(
                    [de_input, encoder.layers[ly_range-i].output])
                de = self.de_block(de, cnn)
            elif i > 3 and i < ly_range-1:  # Last 4 layers don't have droput
                conc = Concatenate()([de, encoder.layers[ly_range-i].output])
                de = self.de_block(conc, cnn, dropout=False)
            elif i == ly_range-1:  # Last layer is tanh activation
                conc = Concatenate()([de, encoder.layers[ly_range-i].output])
                de = self.de_block(conc, cnn, activation='tanh')
            else:
                conc = Concatenate()([de, encoder.layers[ly_range-i].output])
                de = self.de_block(conc, cnn)

        model = keras.Model(inputs=encoder.inputs, outputs=de)
        return model
