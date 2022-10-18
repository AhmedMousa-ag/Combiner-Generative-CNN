
#from ..tensorflow_builder import tf_builder
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalizationV2, Dropout, Input, Concatenate, Lambda, ReLU
from keras.activations import tanh
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy, MeanAbsoluteError
import os
class pix2pix():
    """This model is a simple autoencoder as our base model"""
    def __init__(self,learning_rate=0.0002):
        self.channels = 3
        self.size = 4
        self.strides = 2
        self.padding = "same"
        self.drop_rate = 0.50  # The droput rate in dropout layer
        self.de_num = 0
        self.en_num = 1
        self.learning_rate = learning_rate
        self._build_model()

    def _build_model(self):
        #TODO check initilaizers #initializer = tf.random_normal_initializer(0., 0.02) , (kernel_initializer=initializer, use_bias=False)
        self.generator = self.generator()
        self.discrimnator = self.discrimnator()

    def discrimnator(self):
        # C64-C128-C256-C512
        cnn_list = [128, 256, 512]
        input_x = Input(shape=[256, 256, 3], dtype=tf.float64,name="Input_X")
        input_y = Input(shape=[256, 256, 3], dtype=tf.float64,name="Input_Y")
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
        model = keras.Model(inputs =(input_x,input_y),outputs =output, name="Discriminator_model")
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
        en_cnn_list = [64,128,256,512,512,512,512,512]
        
        inputs = tf.keras.layers.Input(shape=[256, 256, 3])
        x = inputs
        skips = []
        for i,cnn in enumerate(en_cnn_list):
          x = self.en_block(x,cnn)
          skips.append(x)
        skips = reversed(skips[:-1])

        for cnn, skip in zip(de_cnn_list,skips):
          x = self.de_block(x,cnn)
          x = Concatenate()([x,skip])
        
        output= self.de_block(x,self.channels)
        
        model = keras.Model(inputs=inputs, outputs=output,name="Generator_model")
        return model

    def check_point(self,generator_optimizer,discriminator_optimizer,generator,discriminator):
        checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discrimnator

    def discriminator_loss(self,disc_real_pred, disc_generated_pred):
        real_loss = self.bce_loss(tf.ones_like(disc_real_pred), disc_real_pred)

        generated_loss = self.bce_loss(tf.zeros_like(disc_generated_pred), disc_generated_pred)

        total_disc_loss = real_loss + generated_loss
        return total_disc_loss
    
    def generator_loss(self,disc_generated_output, gen_output, target):
        gan_loss = self.bce_loss(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean absolute error
        l1_loss = self.L1_loss(target , gen_output)

        total_gen_loss = gan_loss + (self.l1_lambda * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    @tf.function
    def step(self,input_image, target,generator,generator_optimizer,
                discriminator,discriminator_optimizer):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            y_gen_fake = generator(input_image, training=True)
            dis_pred_real = discriminator((input_image, target), training=True)
            dis_pred_fake = discriminator((input_image, y_gen_fake), training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(dis_pred_fake, y_gen_fake, target)
            disc_loss = self.discriminator_loss(dis_pred_real, dis_pred_fake)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
        return disc_loss,gen_total_loss
    
    def fit(self,dataset,epochs,l1_lambda=100,save_check_point=False):

        generator = self.get_generator()
        discriminator = self.get_discriminator()

        self.bce_loss = BinaryCrossentropy(from_logits=True)
        self.L1_loss = MeanAbsoluteError()
        self.l1_lambda = l1_lambda

        disc_optim = Adam(learning_rate=self.learning_rate,beta_1=0.5,beta_2=0.999)
        gen_optim = Adam(learning_rate=self.learning_rate,beta_1=0.5,beta_2=0.999)
        if save_check_point:
            self.check_point(self,generator_optimizer=gen_optim,discriminator_optimizer=disc_optim,
            generator=generator,discriminator=discriminator)

        for epoch in range(epochs):
            for i,(x_batch_train,y_batch_train) in enumerate(dataset):
                dis_loss_val,gen_loss_val = self.step(input_image=x_batch_train, target=y_batch_train,
                generator=generator,generator_optimizer=gen_optim,
                discriminator=discriminator,discriminator_optimizer=disc_optim)
                print_results(epoch=epoch,step=i,dis_loss=dis_loss_val,gen_loss=gen_loss_val)
            if save_check_point:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)    
                
def print_results(epoch,step,dis_loss,gen_loss,p_every_num_step=10):
    if step % p_every_num_step ==0:
        print(f"Epoch: {epoch}")
        print(f"Step: {step}, Discriminator loss: {dis_loss}, Generator loss: {gen_loss}")

            
