
#from ..tensorflow_builder import tf_builder
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalizationV2, Dropout, Input, Concatenate, Lambda, ReLU
from keras.activations import tanh
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy, MeanAbsoluteError

class __model():
    """This model is a simple autoencoder as our base model"""

    def __init__(self,learning_rate=0.0002, *args, **kwargs):
        self.channels = 3
        self.strides = 1
        self.padding = "same"
        self.drop_rate = 0.50  # The droput rate in dropout layer
        self.de_num = 0
        self.en_num = 1
        self.learning_rate = learning_rate
        self._build_model()

    def _build_model(self):
        self.generator = self.generator()
        self.discrimnator = self.discrimnator()

    def discrimnator(self):
        # C64-C128-C256-C512
        cnn_list = [128, 256, 512]
        input_x = Input(shape=[256, 256, 3], dtype=tf.float64)
        input_y = Input(shape=[256, 256, 3], dtype=tf.float64)
        x = Concatenate()([input_x,input_y])

        x = Conv2D(64, self.channels*2, strides=2,
                         padding=self.padding)(x)
        x = LeakyReLU(0.2)(x)

        for cnn in cnn_list:
            x = Conv2D(cnn, self.channels, strides=self.strides,
                             padding=self.padding)(x)
            x = BatchNormalizationV2()(x)
            x = LeakyReLU(0.2)(x)
        output = Conv2D(1, self.channels, strides=self.strides,
                         padding=self.padding, activation="sigmoid")(x)
        model = keras.Model(inputs = (input_x,input_y),outputs =output )
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

    @tf.function
    def fit(self,dataset,epochs,l1_lambda=100):

        generator = self.generator()
        discriminator = self.discrimnator()

        bce_loss = BinaryCrossentropy(from_logits=True)
        L1_loss = MeanAbsoluteError()
        disc_optim = Adam(learning_rate=self.learning_rate,beta_1=0.5,beta_2=0.999)
        gen_optim = Adam(learning_rate=self.learning_rate,beta_1=0.5,beta_2=0.999)

        for epoch in epochs:
            for step,(x_batch_train,y_batch_train) in enumerate(dataset):
                # training Generator first

                discriminator.trainable=False # We don't want to train the discriminator here
                with tf.GradientTape as tape:
                    y_gen_fake = generator(x_batch_train,training=True)
                    dis_pred_fake = discriminator((x_batch_train,y_gen_fake),training=False)
                    gen_loss_fake = bce_loss(x_batch_train,y_gen_fake)
                    l1 = L1_loss(y_gen_fake,y_batch_train) + l1_lambda
                    gen_loss_val = gen_loss_fake + l1
                
                gen_grads = tape.gradient(gen_loss_val, generator.trainable_weights)
                gen_optim.apply_gradients(zip(gen_grads, generator.trainable_weights))


                # Training Discriminator
                discriminator.trainable=True
                with tf.GradientTape as tape:
                    dis_pred_real = discriminator((x_batch_train,y_batch_train),training=True)
                    dis_loss_real = bce_loss(dis_pred_real,tf.ones_like(dis_pred_real))

                    dis_pred_fake = discriminator((x_batch_train,y_gen_fake),training=True)
                    dis_loss_fake = bce_loss(dis_pred_fake,tf.zeros_like(dis_pred_real))
                    dis_loss_val = (dis_loss_real+dis_loss_fake) / 2

                dis_grads = tape.gradient(dis_loss_val, discriminator.trainable_weights)
                disc_optim.apply_gradients(zip(dis_grads, discriminator.trainable_weights))

                
                


            

                

