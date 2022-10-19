
#from ..tensorflow_builder import tf_builder
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalizationV2, Dropout, Input, Concatenate, Lambda, ReLU
from keras.activations import tanh
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy, MeanAbsoluteError
import os
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from Utils.visualize_images.visualize_images import display_one_image
from datetime import datetime


class dec_lt_per_epoch(LearningRateSchedule):
    "In case we needed to decrease learning rate"
    def __init__(self, initial_learning_rate,decay_at_steps: list,decay=0.0001):
        self.initial_learning_rate = initial_learning_rate
        self.curr_lr = self.initial_learning_rate
        self.decay = decay
        self.decay_at_step = decay_at_steps

    def __call__(self, step):
        if step == self.decay_at_steps[0]:
            if len(self.decay_at_step)>1:
                self.decay_at_step.pop(0)
            self.curr_lr -= self.decay
        return self.curr_lr

class pix2pix():
    """This model is a simple autoencoder as our base model"""
    def __init__(self, learning_rate=0.0002,load_models=False):
        self.output_channel = 3
        self.size = 4
        self.strides = 2
        self.padding = "same"
        self.drop_rate = 0.50  # The droput rate in dropout layer
        self.de_num = 0
        self.en_num = 1
        self.learning_rate = learning_rate
        self.initializer = tf.random_normal_initializer(0., 0.02)
        if not load_models:
            self._build_model()
        else:
            self.load_models()

    def _build_model(self):
        #TODO check initilaizers #initializer = tf.random_normal_initializer(0., 0.02) , (kernel_initializer=initializer, use_bias=False)
        self.generator = self.generator()
        self.discrimnator = self.discrimnator()

    def discrimnator(self):
        # C64-C128-C256-C512
        cnn_list = [128, 256, 512]
        input_x = Input(shape=[256, 256, 3], dtype=tf.float64, name="Input_X")
        input_y = Input(shape=[256, 256, 3], dtype=tf.float64, name="Input_Y")
        x = Concatenate()([input_x, input_y])

        x = Conv2D(64, self.size*2, strides=2,
                   padding=self.padding, kernel_initializer=self.initializer)(x)
        x = LeakyReLU(0.2)(x)

        for cnn in cnn_list:
            x = Conv2D(cnn, self.size, strides=self.strides,
                       padding=self.padding, kernel_initializer=self.initializer)(x)
            x = BatchNormalizationV2()(x)
            x = LeakyReLU(0.2)(x)
        output = Conv2D(1, self.size, strides=self.strides,
                        padding=self.padding, kernel_initializer=self.initializer, activation="sigmoid")(x)
        model = keras.Model(inputs=(input_x, input_y),
                            outputs=output, name="Discriminator_model")
        return model

    def en_block(self, input, cnn, batch_norm=True):
        self.en_num += 1
        x = Conv2D(cnn, self.size, strides=self.strides,
                   padding=self.padding, kernel_initializer=self.initializer, name=f"Encoder_Conv_{cnn}_{self.en_num}")(input)
        if batch_norm:
            x = BatchNormalizationV2(
                name=f"Encoder_BatchNorm_{cnn}_{self.en_num}")(x)
        x = LeakyReLU(0.2, name=f"Encoder_LRelu_{cnn}_{self.en_num}")(x)
        return x

    def de_block(self, input, cnn, dropout=True, activation=None):
        # CD512-CD512-CD512-C512-C256-C128-C64
      self.de_num += 1
      x = Conv2DTranspose(cnn, self.size, strides=self.strides,
                          padding=self.padding, kernel_initializer=self.initializer, name=f"Decoder_Conv_{cnn}_{self.de_num}")(input)
      if activation:    # Last layer is tanh activation,
        # if we call any activation it will call this layer which is lambda layer for tanh activation
        # And doesn't have Dropout layer
        x = Lambda(lambda x: tanh(x), name=f"Decoder_tanh_{cnn}_{self.de_num}")(
            x)
      else:
        x = BatchNormalizationV2(
            name=f"Decoder_BatchNorm_{cnn}_{self.de_num}")(x)
        x = ReLU(0.2, name=f"Decoder_LRelu_{cnn}_{self.de_num}")(x)
        if dropout:
            x = Dropout(self.drop_rate,
                        name=f"Decoder_Dropout_{cnn}_{self.de_num}")(x)
      return x

    def generator(self):
        de_cnn_list = [512, 512, 512, 512, 256, 128, 64]  # As in paper
        en_cnn_list = [64, 128, 256, 512, 512, 512, 512, 512]

        inputs = tf.keras.layers.Input(shape=[256, 256, 3])
        x = inputs
        skips = []
        for i, cnn in enumerate(en_cnn_list):
            if i == 0:
                x = self.en_block(x, cnn, batch_norm=False)
            else:
                x = self.en_block(x, cnn)
            skips.append(x)

        skips = reversed(skips[:-1])
        i = 0
        for cnn, skip in zip(de_cnn_list, skips):
            if i <= 2:
                x = self.de_block(x, cnn, dropout=True)
                x = Concatenate()([x, skip])
            else:
                x = self.de_block(x, cnn)
                x = Concatenate()([x, skip])
            i += 1
        output = self.de_block(x, self.output_channel, activation="tanh")

        model = keras.Model(inputs=inputs, outputs=output,
                            name="Generator_model")
        return model

    def check_point(self, generator_opt, discriminator_opt, generator, discriminator):
        checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=generator_opt,
                                              discriminator_optimizer=discriminator_opt,
                                              generator=generator,
                                              discriminator=discriminator)

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discrimnator

    def save_generator(self,path:str,h5=False):
        gen_path = "generator" if not h5 else "generator.h5"
        save_path  = os.join(path,gen_path)
        self.generator.save(save_path)

    def save_discriminator(self,path:str,h5=False):
        dis_path = "discriminator" if not h5 else "discriminator.h5"
        save_path  = os.join(path,dis_path)
        self.discrimnator.save(save_path)

    def save_models(self,path:str,h5=False):
        self.save_discriminator(path=path,h5=h5)
        self.save_discriminator(path=path,h5=h5)

    def save_generator(self,path:str,h5=False):
        gen_path = "generator" if not h5 else "generator.h5"
        gen_save_path  = os.join(path,gen_path)
        self.generator = tf.keras.models.load_model(gen_save_path)

    def load_discrimnator(self,path:str,h5=False):
        dis_path = "discriminator" if not h5 else "discriminator.h5"
        dis_save_path  = os.join(path,dis_path)
        self.discrimnator =  tf.keras.models.load_model(dis_save_path)

    def load_models(self,path:str,h5=False):
        self.load_discrimnator(path=path,h5=h5)
        self.load_generator(path=path,h5=h5)
        

    def discriminator_loss(self, disc_real_pred, disc_generated_pred):
        real_loss = self.bce_loss(tf.ones_like(disc_real_pred), disc_real_pred)

        generated_loss = self.bce_loss(tf.zeros_like(
            disc_generated_pred), disc_generated_pred)
        #Divide by 2 as we want it to learn slower than the generator
        total_disc_loss = (real_loss + generated_loss)/2  #TODO remove it if it doesn't work
        return total_disc_loss

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.bce_loss(tf.ones_like(
            disc_generated_output), disc_generated_output)

        # Mean absolute error
        l1_loss = self.L1_loss(target, gen_output)

        total_gen_loss = gan_loss + (self.l1_lambda * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    @tf.function
    def step(self, input_image, target, generator, generator_optimizer,
             discriminator, discriminator_optimizer):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            y_gen_fake = generator(input_image, training=True)
            dis_pred_real = discriminator((input_image, target), training=True)
            dis_pred_fake = discriminator(
                (input_image, y_gen_fake), training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(
                dis_pred_fake, y_gen_fake, target)
            disc_loss = self.discriminator_loss(dis_pred_real, dis_pred_fake)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))
        return disc_loss, gen_total_loss

    def fit(self, dataset, epochs, l1_lambda=100, save_models=True,save_interval=100,test_image: str =None,generate_image_interval=50):
        #TODO make a reduce learning rate function or callback
        generator = self.get_generator()
        discriminator = self.get_discriminator()

        self.bce_loss = BinaryCrossentropy(from_logits=True)
        self.L1_loss = MeanAbsoluteError()
        self.l1_lambda = l1_lambda

        disc_optim = Adam(learning_rate=self.learning_rate,
                          beta_1=0.5, beta_2=0.999)
        gen_optim = Adam(learning_rate=self.learning_rate,
                         beta_1=0.5, beta_2=0.999)
        
        if save_models:
            time_stamp = datetime.timestamp(datetime.now())

        for epoch in range(epochs):
            for i, (x_batch_train, y_batch_train) in enumerate(dataset):
                dis_loss_val, gen_loss_val = self.step(input_image=x_batch_train, target=y_batch_train,
                                                       generator=generator, generator_optimizer=gen_optim,
                                                       discriminator=discriminator, discriminator_optimizer=disc_optim)
                print_results(epoch=epoch, step=i,
                              dis_loss=dis_loss_val, gen_loss=gen_loss_val)
            
            if save_models:
                if epoch%save_interval == 0:
                    self.save_models(path=f'{time_stamp}training_saved')
            if test_image:
                if epoch % generate_image_interval == 0:
                    display_one_image(generator(test_image),"Generated Image") #To see how it goes every 50 epochs


def print_results(epoch, step, dis_loss, gen_loss, p_every_num_step=10):
    if step % p_every_num_step == 0:
        print(f"Epoch: {epoch}")
        print(
            f"Step: {step}, Discriminator loss: {dis_loss}, Generator loss: {gen_loss}")
