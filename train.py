import tensorflow as tf
import numpy as np 
import os 
import cv2 

from utils import progressBar
from dcgan import Generator, Discriminator
from tensorflow.keras import optimizers
                  
        
class Trainer():
    def __init__(self, 
                 progress_dir,
                 checkpoint_dir,
                 z_dim=100, 
                 test_size=4,
                 batch_size=100,
                 learning_rate=0.0002,
                 beta_1=0.5):
                
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.test_size = test_size
        self.progress_dir = progress_dir
        self.test_points = tf.random.normal(shape=(test_size**2, z_dim))
        
        self.ckpt = tf.train.Checkpoint(epoch=tf.Variable(1),
                                        val_loss=tf.Variable(np.inf),
                                        gan_optimizer=optimizers.Adam(lr=learning_rate, beta_1=beta_1),
                                        dis_optimizer=optimizers.Adam(lr=learning_rate, beta_1=beta_1),
                                        generator=Generator(z_dim),
                                        discriminator=Discriminator())
            
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint=self.ckpt,
                                                       directory=checkpoint_dir,
                                                       max_to_keep=5)
        self.restore_checkpoint()
        
        self.gen_loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.dis_loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.gen_metric = tf.metrics.Mean()
        self.dis_metric = tf.metrics.Mean()
        
        
    def restore_checkpoint(self):
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print(f"Model restored at epoch {self.ckpt.epoch.numpy()}")
            
    @tf.function 
    def discriminator_train_step(self, noises, real_images):
        with tf.GradientTape() as tape:
            fake_images = self.ckpt.generator(noises, training=True)
            
            real_labels = self.ckpt.discriminator(real_images, training=True)
            fake_labels = self.ckpt.discriminator(fake_images, training=True)
            
            real_loss = self.dis_loss_fn(tf.ones(shape=(self.batch_size)), real_labels)
            fake_loss = self.dis_loss_fn(tf.zeros(shape=(self.batch_size)), fake_labels)
            
            dis_loss = real_loss + fake_loss
            
        dis_gradients = tape.gradient(dis_loss, 
                                      self.ckpt.discriminator.trainable_variables)
        self.ckpt.dis_optimizer.apply_gradients(zip(dis_gradients, 
                                                    self.ckpt.discriminator.trainable_variables))
        self.dis_metric.update_state(dis_loss)
        return dis_loss
        
    @tf.function 
    def generator_train_step(self, noises):
        with tf.GradientTape() as tape:
            fake_labels = self.ckpt.discriminator(self.ckpt.generator(noises, 
                                                                      training=True), 
                                                  training=True)
            gen_loss = self.gen_loss_fn(tf.ones(shape=(self.batch_size)), 
                                        fake_labels)
            
        gen_gradients = tape.gradient(gen_loss, 
                                      self.ckpt.generator.trainable_variables)
        self.ckpt.gan_optimizer.apply_gradients(zip(gen_gradients, 
                                                    self.ckpt.generator.trainable_variables))
        self.gen_metric.update_state(gen_loss)
        return gen_loss
        
    def train_loop(self, dataset, epochs, total_steps):
        for epoch in range(epochs):
            for i, real_images in enumerate(dataset):
        
                noises = tf.random.normal(shape=(self.batch_size, self.z_dim))
                dis_loss = self.discriminator_train_step(noises, real_images)
                gen_loss = self.generator_train_step(noises)
                
                progressBar(epoch+1, i+1, total_steps, dis_loss.numpy(), gen_loss.numpy())
        
                if (i+1) % 50 == 0:
                    self.generate_training_progress_result(self.ckpt.generator, epoch+1, i+1)
            
            dis_loss_epoch = self.dis_metric.result().numpy()
            gen_loss_epoch = self.gen_metric.result().numpy()
            
            print()
            print(f"Epoch {epoch+1} - D-Loss: {dis_loss_epoch} - G-Loss: {gen_loss_epoch}")
            
            self.ckpt_manager.save()
            self.ckpt.epoch.assign_add(1)
            self.dis_metric.reset_states()
            self.gen_metric.reset_states()
            
    def generate_training_progress_result(self, model, epoch, step):
        
        test_images = model(self.test_points)
        test_images = (test_images * 127.5) + 127.5
        test_images = test_images.numpy().astype("uint8")
        
        _, height, width, depth = test_images.shape
        
        test_images = test_images.reshape(self.test_size, 
                                          self.test_size, 
                                          height, 
                                          width, 
                                          depth)
        test_images = test_images.transpose(0,2,1,3,4)
        test_images = test_images.reshape(height*self.test_size,
                                          width*self.test_size,
                                          depth)
        path = os.path.join(self.progress_dir, f"Epoch_{epoch}_on_{step}_batch.png")
        cv2.imwrite(path, test_images[...,::-1])