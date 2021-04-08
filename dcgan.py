import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model

########## Generator ##########

class DeconvBlock(layers.Layer):
    def __init__(self, weight_init, channels, momentum=0.9, epsilon=1e-5):
        super(DeconvBlock, self).__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        self.channels = channels
        self.weight_init = weight_init
        
    def build(self, input_shape):
        self.deconv = layers.Conv2DTranspose(self.channels,
                                             kernel_size=5,
                                             strides=2, 
                                             padding="same",
                                             kernel_initializer=self.weight_init)
        self.batchnorm = layers.BatchNormalization(momentum=self.momentum,
                                                   epsilon=self.epsilon)
        super(DeconvBlock, self).build(input_shape)
        
    def call(self, inputs, training=False):
        x = self.deconv(inputs)
        x = self.batchnorm(x, training=training)
        x = layers.ReLU()(x)
        return x
        
    def get_config(self):
        config = super(DeconvBlock, self).get_config()
        config.update({
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "channels": self.channels
        })
        return config
        

class Generator(Model):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.weight_init = tf.keras.initializers.RandomNormal(stddev=0.02)
        self.channel_list = [1024, 512, 256, 128]
        
    def build(self, input_shape):
        self.linear = layers.Dense(self.channel_list[0]*4*4,
                                    kernel_initializer=self.weight_init,
                                    activation="relu")

        self.upblocks = [DeconvBlock(self.weight_init, c) for c in self.channel_list[1:]] 
        self.deconv = layers.Conv2DTranspose(3, 
                                             kernel_size=5,
                                             strides=2,
                                             padding="same",
                                             kernel_initializer=self.weight_init,
                                             activation="tanh")
        self.shape = input_shape
        super(Generator, self).build(input_shape)
    
    def call(self, inputs, training=False):
        x = self.linear(inputs)
        x = layers.Reshape(target_shape=(4,4,self.channel_list[0]))(x)
        for upblock in self.upblocks:
            x = upblock(x, training)
        x = self.deconv(x)
        return x 
    
    def summary(self):
        inputs = layers.Input(shape=self.shape[1:])
        outputs = self.call(inputs)
        model = tf.keras.Model(inputs, outputs)
        model.summary()
        
########## Discriminator ##########

class ConvBlock(layers.Layer):
    def __init__(self, weight_init, channels, momentum=0.9, epsilon=1e-5):
        super(ConvBlock, self).__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        self.channels = channels
        self.weight_init = weight_init
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(self.channels,
                                  kernel_size=5,
                                  strides=2, 
                                  padding="same",
                                  kernel_initializer=self.weight_init)
        self.batchnorm = layers.BatchNormalization(momentum=self.momentum,
                                                   epsilon=self.epsilon)
        super(ConvBlock, self).build(input_shape)
        
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.batchnorm(x, training=training)
        x = layers.LeakyReLU(0.2)(x)
        return x
        
    def get_config(self):
        config = super(ConvBlock, self).get_config()
        config.update({
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "channels": self.channels
        })
        return config
    
        
class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.weight_init = tf.keras.initializers.RandomNormal(stddev=0.02)
        self.channel_list = [32, 64, 128, 256, 512]
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(self.channel_list[0],
                                  kernel_size=5,
                                  strides=2,
                                  padding="same",
                                  kernel_initializer=self.weight_init)
        
        self.downblocks = [ConvBlock(self.weight_init, c) for c in self.channel_list[1:]] 
        self.linear = layers.Dense(1, activation="sigmoid")
        
        self.shape = input_shape
        super(Discriminator, self).build(input_shape)
    
    def call(self, x, training=False):
        x = self.conv(x)
        x = layers.LeakyReLU(0.2)(x)
        for downblock in self.downblocks:
            x = downblock(x, training)
        x = layers.Flatten()(x)
        x = self.linear(x)
        return x 
    
    def summary(self):
        inputs = layers.Input(shape=self.shape[1:])
        outputs = self.call(inputs)
        model = tf.keras.Model(inputs, outputs)
        model.summary()