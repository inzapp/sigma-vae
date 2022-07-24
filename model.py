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
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Model:
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.vae = None
        self.decoder = None

    def build(self):
        assert self.input_shape[0] % 32 == 0
        assert self.input_shape[1] % 32 == 0
        encoder_input, encoder_output = self.build_encoder()
        decoder_input, decoder_output = self.build_decoder()
        z, mu, log_var = encoder_output
        self.decoder = tf.keras.models.Model(decoder_input, decoder_output)
        vae_output = self.decoder(z)
        self.vae = tf.keras.models.Model(encoder_input, [vae_output, mu, log_var])
        return self.vae, self.decoder

    def build_encoder(self):
        encoder_input = tf.keras.layers.Input(shape=self.input_shape)
        x = encoder_input
        x = self.conv2d(x, 16,  3, 2, activation='relu', bn=True)
        x = self.conv2d(x, 32,  3, 2, activation='relu', bn=True)
        x = self.conv2d(x, 64,  3, 2, activation='relu', bn=True)
        x = self.conv2d(x, 128, 3, 2, activation='relu', bn=True)
        x = self.conv2d(x, 256, 3, 2, activation='relu', bn=True)
        x = self.flatten(x)
        x = self.dense(x, 4096, activation='relu', bn=True)
        mu = self.dense(x, self.latent_dim, activation='linear', bn=True)
        log_var = self.dense(x, self.latent_dim, activation='linear', bn=True)
        z = self.sampling(mu, log_var)
        return encoder_input, [z, mu, log_var]

    def build_decoder(self):
        target_rows = self.input_shape[0] // 32
        target_cols = self.input_shape[1] // 32
        target_channels = 256

        decoder_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = decoder_input
        x = self.dense(x, 4096, activation='relu', bn=True)
        x = self.dense(x, target_rows * target_cols * target_channels, activation='relu', bn=True)
        x = self.reshape(x, (target_rows, target_cols, target_channels))
        x = self.conv2d_transpose(x, 256, 3, 2, activation='relu', bn=True)
        x = self.conv2d_transpose(x, 128, 3, 2, activation='relu', bn=True)
        x = self.conv2d_transpose(x, 64,  3, 2, activation='relu', bn=True)
        x = self.conv2d_transpose(x, 32,  3, 2, activation='relu', bn=True)
        x = self.conv2d_transpose(x, 16,  3, 2, activation='relu', bn=True)
        decoder_output = self.conv2d_transpose(x, self.input_shape[-1], 1, 1, activation='sigmoid')
        return decoder_input, decoder_output

    def sampling(self, mu, log_var):
        def function(args):
            mu, log_var = args
            batch = K.shape(mu)[0]
            dim = K.shape(mu)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return mu + K.exp(log_var * 0.5) * epsilon
        return tf.keras.layers.Lambda(function=function)([mu, log_var])

    def conv2d(self, x, filters, kernel_size, strides=1, bn=True, activation='relu', alpha=0.2):
        x = tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            use_bias=False if bn else True,
            kernel_initializer='he_normal')(x)
        if bn:
            x = tf.keras.layers.BatchNormalization()(x)
        return self.activation(x, activation)

    def conv2d_transpose(self, x, filters, kernel_size, strides=1, bn=True, activation='relu', alpha=0.2):
        x = tf.keras.layers.Conv2DTranspose(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            use_bias=False if bn else True,
            kernel_initializer='he_normal')(x)
        if bn:
            x = tf.keras.layers.BatchNormalization()(x)
        return self.activation(x, activation)

    def dense(self, x, units, bn=True, activation='relu', alpha=0.2):
        x = tf.keras.layers.Dense(
            units=units,
            use_bias=False if bn else True,
            kernel_initializer='he_normal')(x)
        if bn:
            x = tf.keras.layers.BatchNormalization()(x)
        return self.activation(x, activation)

    def activation(self, x, activation, alpha=0.2):
        if activation == 'leaky':
            x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)
        else:
            x = tf.keras.layers.Activation(activation=activation)(x)
        return x

    def reshape(self, x, target_shape):
        return tf.keras.layers.Reshape(target_shape=target_shape)(x)

    def flatten(self, x):
        return tf.keras.layers.Flatten()(x)

    def summary(self):
        self.decoder.summary()
        print()
        self.vae.summary()

