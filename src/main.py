from keras import backend as backend
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation
from keras.models import Sequential, Model, clone_model
from keras.models import load_model, save_model
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizer_v2.adam import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.image import resize

import matplotlib.pyplot as plt
import sys
import os
from google.colab.patches import cv2_imshow
import numpy as np
from time import time
import cv2

backend.set_image_data_format('channels_last')


class Dataset:
  def __init__(self, directory: str, start_folder: int = 0):
    self.__directory: str = directory
    self.content: np.array = np.array([])
    self.currFolder: int = start_folder
    self.ReloadData()

  def ReloadData(self)-> None:
    self.currFolder += 1
    prev = time()
    c = 0
    folder = os.listdir(self.__directory)[self.currFolder]
    content = []
    for file in os.listdir(self.__directory + folder + "/"):
      img = load_img(self.__directory + folder + "/" + file)
      inp_arr = img_to_array(img)
      input_arr = np.array(inp_arr).astype("float32")
      input_arr = (input_arr - 127.5) / 127.5
      content.append(input_arr)
      c += 1
      if time() - prev >= 1.0:
        print(f"Reloading dataset: {float('{:.3f}'.format((c / 1000) * 100))}%")
        prev = time() 
      
    self.content = np.array(content)
    
  def GetRandomBatch(self, batch_size: int)->np.array:
    return np.array([self.content[i] for i in 
                     np.random.randint(0, self.GetFilesCount(), batch_size)])

  def GetFilesCount(self)->int:
    return self.content.shape[0]

class Gan:
    def __init__(self, dataset: Dataset, img_rows: int = 32, img_cols: int = 32,
                 img_channels: int = 3, latent_dim: int = 500,
                 optimizer=Adam(learning_rate=0.0001, beta_1=0.5),
                 loadModel: bool = False, modelPath: str = ""):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.latent_dim = latent_dim
        self.optimizer = optimizer
        self.dataset = dataset
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.discriminator = self.build_discriminator() \
                                if not loadModel \
                                else load_model(modelPath + "discriminator/")

        self.discriminator.trainable = False
        self.generator = self.build_generator() \
                                if not loadModel \
                                else load_model(modelPath + "generator/")

        self.gan = self.build_combined() \
                                 if not loadModel \
                                 else load_model(modelPath + "gan/")

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(64, kernel_size=(3, 3),
                         padding='same', input_shape=self.img_shape))
        
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(512, kernel_size=(3, 3),
                         strides=(2, 2), padding='same'))
        
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(512, kernel_size=(3, 3),
                         strides=(2, 2), padding='same'))
        
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(512, kernel_size=(3, 3),
                         strides=(2, 2), padding='same'))
        
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(256, kernel_size=(3, 3),
                         strides=(2, 2), padding='same'))
        
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=(3, 3),
                         strides=(2, 2), padding='same'))
        
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=self.optimizer,
                      loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def build_generator(self):
        model = Sequential()
        n_nodes = 256 * 4 * 4

        model.add(Dense(n_nodes, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 256)))

        model.add(Conv2DTranspose(512, kernel_size=(4, 4),
                                  strides=(2, 2), padding='same')) # 8x8
        
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(512, kernel_size=(4, 4),
                                  strides=(2, 2), padding='same')) # 16#16
        
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(512, kernel_size=(4, 4),
                                  strides=(2, 2), padding='same')) # 32X32
        
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(256, kernel_size=(4, 4),
                                  strides=(2, 2), padding='same')) # 64x64
        
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(128, kernel_size=(4, 4),
                                  strides=(2, 2), padding='same')) # 128x128
        
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(self.img_channels, kernel_size=(3, 3),
                         activation='tanh', padding='same')) # 128x128 
                                                             # strides = (1,1)
        return model

    def build_combined(self):
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        model.summary()
        return model

    def generate_real(self, amount):
        X = self.dataset.GetRandomBatch(amount)
        X = X.reshape((amount, *self.img_shape))
        y = np.ones((amount, 1))
        return X, y

    def generate_latent_points(self, amount):
        x_input = np.random.randn(self.latent_dim * amount)
        x_input = x_input.reshape(amount, self.latent_dim)
        return x_input

    def generate_fake(self, amount):
        x_input = self.generate_latent_points(amount)
        X = self.generator.predict(x_input)
        y = np.zeros((amount, 1))
        return X, y

    def save_pics(self, epoch, n: int):
        examples, _ = self.generate_fake(n*n)
        examples += 1
        examples /= 2.0
        print(examples.shape)

        for i in range(n*n):
            plt.subplot(n, n, 1+i)
            plt.axis('off')
            plt.imshow(examples[i])
        filename = f'/content/drive/MyDrive/GAN faces/new2/Generated sample on epoch {epoch}'
        plt.savefig(filename)
        plt.close()

    def save(self):
      save_model(self.gan, "modelsv2/gan")
      save_model(self.discriminator, "modelsv2/discriminator")
      save_model(self.generator, "modelsv2/generator")

    def summarize(self, epoch, amount=150):
        X_real, y_real = self.generate_real(amount)

        _, accuracy_real = self.discriminator.evaluate(X_real, y_real,
                                                       verbose=0)

        X_fake, y_fake = self.generate_fake(amount)

        _, accuracy_fake = self.discriminator.evaluate(X_fake, y_fake,
                                                       verbose=0)
        print(f'Epoch {epoch}: ' +
              f'Accuracy real: {accuracy_real}, ' +
              f'fake: {accuracy_fake}')

    def train(self, epochs: int = 100, batch_size: int = 32):
        batch_per_epoch = 50
        half_batch = int(batch_size / 2)
        for i in range(1, epochs+1):
            for j in range(1, batch_per_epoch+1):
                X_real, y_real = self.generate_real(half_batch)
                

                d_loss1, _ = self.discriminator.train_on_batch(X_real, y_real)

                X_fake, y_fake = self.generate_fake(half_batch)

                d_loss2, _ = self.discriminator.train_on_batch(X_fake, y_fake)

                X_gan = self.generate_latent_points(batch_size)
                y_gan = np.ones((batch_size, 1))

                g_loss = self.gan.train_on_batch(X_gan, y_gan)
                if j % 5 == 0:
                    os.system("cls")
                    print(f'Epoch : {i} / {epochs}, ' +
                          f'{j} / {batch_per_epoch}, ' +
                          f'd1 = {d_loss1}, d2 = {d_loss2}, g = {g_loss}')
            
            if i % 1 == 0:
              self.save_pics(i, 2)
              self.save()
            if i % 30 == 0:
              self.dataset.ReloadData()
            self.summarize(i)


if __name__ == '__main__':
    dataset = Dataset("/content/drive/MyDrive/thumbnails128x128/",
                      start_folder=30)
    gan = Gan(dataset=dataset, img_cols=128, img_rows=128, img_channels=3)
    gan.train(4900)
