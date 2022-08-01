"""
Authors : inzapp

Github url : https://github.com/inzapp/vae

Copyright 2022 inzapp Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"),
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import natsort
import numpy as np
import tensorflow as tf

from cv2 import cv2
from glob import glob
from tqdm import tqdm
from time import time
from model import Model
from lr_scheduler import LRScheduler
from generator import DataGenerator


class VariationalAutoEncoder:
    def __init__(self,
                 train_image_path=None,
                 input_shape=(64, 64, 1),
                 lr=0.001,
                 batch_size=32,
                 latent_dim=32,
                 iterations=100000,
                 view_grid_size=4,
                 pretrained_model_path='',
                 checkpoint_path='checkpoints',
                 training_view=False):
        assert input_shape[-1] in [1, 3]
        self.lr = lr
        self.iterations = iterations
        self.training_view = training_view
        self.live_view_previous_time = time()
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.checkpoint_path = checkpoint_path
        self.view_grid_size = view_grid_size
        if self.latent_dim == -1:
            self.latent_dim = self.input_shape[0] // 32 * self.input_shape[1] // 32 * 256

        self.model = Model(input_shape=input_shape, latent_dim=self.latent_dim)
        self.vae, self.decoder = self.model.build()
        # if os.path.exists(pretrained_model_path) and os.path.isfile(pretrained_model_path):
        #     print(f'\npretrained model path : {[pretrained_model_path]}')
        #     self.decoder = tf.keras.models.load_model(pretrained_model_path, compile=False)
        #     print(f'input_shape : {self.input_shape}')

        self.train_image_paths = self.init_image_paths(train_image_path)
        self.train_data_generator = DataGenerator(
            image_paths=self.train_image_paths,
            input_shape=input_shape,
            batch_size=batch_size,
            latent_dim=self.latent_dim)
        self.lr_scheduler = LRScheduler(lr=self.lr, iterations=self.iterations)

    def init_image_paths(self, image_path):
        return glob(f'{image_path}/**/*.jpg', recursive=True)

    @tf.function
    def compute_gradient(self, model, optimizer, x, y_true):
        with tf.GradientTape() as tape:
            y_pred, mu, log_var = model(x, training=True)
            mu_mean = tf.reduce_mean(mu)
            log_var_mean = tf.reduce_mean(log_var)
            reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            balancing_ratio = (tf.cast(tf.shape(y_true)[1] + tf.shape(y_true)[2], dtype=tf.float32) * 0.5) / 32.0
            balancing_factor = tf.pow(balancing_ratio, balancing_ratio) * 1024.0
            kl_weight = (1.0 / balancing_factor) * tf.cast(tf.shape(mu)[-1], dtype=tf.float32)
            kl_divergence = -0.5 * (1.0 + log_var - tf.square(mu) - tf.exp(log_var))
            kl_loss = tf.reduce_mean(kl_divergence) * kl_weight
            kl_divergence_mean = tf.reduce_mean(kl_divergence)
            loss = reconstruction_loss + kl_loss
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return reconstruction_loss, kl_divergence_mean, mu_mean, log_var_mean

    def fit(self):
        self.model.summary()
        print(f'\ntrain on {len(self.train_image_paths)} samples.')
        print('start training')
        iteration_count = 0
        optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        while True:
            for batch_x in self.train_data_generator:
                self.lr_scheduler.schedule_step_decay(optimizer, iteration_count)
                reconstruction_loss, kl_loss, mu, log_var = self.compute_gradient(self.vae, optimizer, batch_x, batch_x)
                iteration_count += 1
                print(f'[iteration count : {iteration_count:6d}] reconstruction_loss : {reconstruction_loss:.4f}, kl_loss : {kl_loss:.4f}, mu : {mu:.4f}, log_var : {log_var:.4f}')
                if self.training_view:
                    self.training_view_function()
                if iteration_count % 1000 == 0:
                    model_path_without_extention = f'{self.checkpoint_path}/decoder_{iteration_count}_iter' 
                    self.decoder.save(f'{model_path_without_extention}.h5', include_optimizer=False)
                    generated_images = self.get_generated_images(grid_size=25)
                    cv2.imwrite(f'{model_path_without_extention}.jpg', generated_images)
                    print(f'[iteration count : {iteration_count:6d}] model with generated images saved with {model_path_without_extention} h5 and jpg\n')
                if iteration_count == self.iterations:
                    print('\n\ntrain end successfully')
                    while True:
                        decoded_images = self.get_decoded_images()
                        generated_images = self.get_generated_images(grid_size=self.view_grid_size)
                        cv2.imshow('decoded_images', decoded_images)
                        cv2.imshow('generated_images', generated_images)
                        key = cv2.waitKey(0)
                        if key == 27:
                            exit(0)

    @tf.function
    def graph_forward(self, model, x):
        with tf.device('/cpu:0'):
            return model(x, training=False)

    def generate_random_image(self, size=1):
        z = np.asarray([DataGenerator.get_z_vector(size=self.latent_dim) for _ in range(size)]).astype('float32')
        y = np.asarray(self.graph_forward(self.decoder, z))
        y = DataGenerator.denormalize(y)
        generated_images = np.clip(np.asarray(y).reshape((size,) + self.input_shape), 0.0, 255.0).astype('uint8')
        return generated_images[0] if size == 1 else generated_images

    def generate_latent_space_2d(self, split_size=10):
        assert split_size > 1
        assert self.latent_dim == 2
        space = np.linspace(-1.0, 1.0, split_size)
        z = []
        for i in range(split_size):
            for j in range(split_size):
                z.append([space[i], space[j]])
        z = np.asarray(z).reshape((split_size * split_size, 2)).astype('float32')
        y = np.asarray(self.graph_forward(self.decoder, z))
        y = DataGenerator.denormalize(y)
        generated_images = np.clip(np.asarray(y).reshape((split_size * split_size,) + self.input_shape), 0.0, 255.0).astype('uint8')
        return generated_images

    def predict(self, img):
        img = DataGenerator.resize(img, (self.input_shape[1], self.input_shape[0]))
        x = np.asarray(img).reshape((1,) + self.input_shape).astype('float32')
        x = DataGenerator.normalize(x)
        y = np.asarray(self.graph_forward(self.vae, x)[0]).reshape(self.input_shape)
        y = DataGenerator.denormalize(y)
        decoded_img = np.clip(y, 0.0, 255.0).astype('uint8')
        return img, decoded_img

    def show_interpolation(self, frame=100):
        space = np.linspace(-1.0, 1.0, frame)
        for val in space:
            z = np.zeros(shape=(1, self.latent_dim), dtype=np.float32) + val
            y = np.asarray(self.graph_forward(self.decoder, z))[0]
            y = DataGenerator.denormalize(y)
            generated_image = np.clip(np.asarray(y).reshape(self.input_shape), 0.0, 255.0).astype('uint8')
            cv2.imshow('interpolation', generated_image)
            key = cv2.waitKey(10)
            if key == 27:
                break

    def make_border(self, img, size=5):
        return cv2.copyMakeBorder(img, size, size, size, size, None, value=(192, 192, 192)) 

    def training_view_function(self):
        cur_time = time()
        if cur_time - self.live_view_previous_time > 3.0:
            decoded_images = self.get_decoded_images()
            generated_images = self.get_generated_images(grid_size=self.view_grid_size)
            cv2.imshow('decoded_images', decoded_images)
            cv2.imshow('generated_images', generated_images)
            cv2.waitKey(1)
            self.live_view_previous_time = cur_time

    def get_decoded_images(self):
        img_paths = np.random.choice(self.train_image_paths, size=self.view_grid_size, replace=False)
        input_shape = self.vae.input_shape[1:]
        decoded_images_cat = None
        for img_path in img_paths:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE if input_shape[-1] == 1 else cv2.IMREAD_COLOR)
            img, output_image= self.predict(img)
            img = DataGenerator.resize(img, (input_shape[1], input_shape[0]))
            img, output_image = self.make_border(img), self.make_border(output_image)
            if self.input_shape[-1] == 1:
                img = img.reshape(img.shape + (1,))
                output_image = output_image.reshape(output_image.shape + (1,))
            imgs = np.concatenate([img, output_image], axis=1)
            if decoded_images_cat is None:
                decoded_images_cat = imgs
            else:
                decoded_images_cat = np.append(decoded_images_cat, imgs, axis=0)
        return decoded_images_cat

    def get_generated_images(self, grid_size):
        generated_images_cat = None
        if self.latent_dim == 2:
            generated_images = self.generate_latent_space_2d(split_size=grid_size)
        else:
            generated_images = self.generate_random_image(size=grid_size * grid_size)
        for i in range(grid_size):
            line = None
            for j in range(grid_size):
                generated_image = self.make_border(generated_images[i * grid_size + j])
                if line is None:
                    line = generated_image
                else:
                    line = np.append(line, generated_image, axis=1)
            if generated_images_cat is None:
                generated_images_cat = line
            else:
                generated_images_cat = np.append(generated_images_cat, line, axis=0)
        return generated_images_cat

    def show_generated_images(self):
        while True:
            generated_images = self.get_generated_images(grid_size=self.view_grid_size)
            cv2.imshow('generated_images', generated_images)
            key = cv2.waitKey(0)
            if key == 27:
                break

