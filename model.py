"""
Authors : inzapp

Github url : https://github.com/inzapp/vae

Copyright (c) 2022 Inzapp

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
        x = self.conv2d(x,  16, 3, 2, activation='relu')
        x = self.conv2d(x,  32, 3, 1, activation='relu')
        x = self.conv2d(x,  32, 3, 2, activation='relu')
        x = self.conv2d(x,  64, 3, 1, activation='relu')
        x = self.conv2d(x,  64, 3, 2, activation='relu')
        x = self.conv2d(x, 128, 3, 1, activation='relu')
        x = self.conv2d(x, 128, 3, 2, activation='relu')
        x = self.conv2d(x, 256, 3, 1, activation='relu')
        x = self.conv2d(x, 256, 3, 2, activation='relu')
        x = self.flatten(x)
        mu = self.dense(x, self.latent_dim, activation='linear')
        log_var = self.dense(x, self.latent_dim, activation='linear')
        z = self.sampling(mu, log_var)
        return encoder_input, [z, mu, log_var]

    def build_decoder(self):
        target_rows = self.input_shape[0] // 32
        target_cols = self.input_shape[1] // 32
        target_channels = 256

        decoder_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = decoder_input
        x = self.dense(x, target_rows * target_cols * target_channels, activation='relu')
        x = self.reshape(x, (target_rows, target_cols, target_channels))
        x = self.conv2d_transpose(x, 256, 3, 2, activation='relu')
        x = self.conv2d_transpose(x, 256, 3, 1, activation='relu')
        x = self.conv2d_transpose(x, 128, 3, 2, activation='relu')
        x = self.conv2d_transpose(x, 128, 3, 1, activation='relu')
        x = self.conv2d_transpose(x,  64, 3, 2, activation='relu')
        x = self.conv2d_transpose(x,  64, 3, 1, activation='relu')
        x = self.conv2d_transpose(x,  32, 3, 2, activation='relu')
        x = self.conv2d_transpose(x,  32, 3, 1, activation='relu')
        x = self.conv2d_transpose(x,  16, 3, 2, activation='relu')
        decoder_output = self.conv2d_transpose(x, self.input_shape[-1], 1, 1, kernel_initializer='glorot_normal', activation='sigmoid')
        return decoder_input, decoder_output

    def sampling(self, mu, log_var):
        def function(args):
            mu, log_var = args
            batch = K.shape(mu)[0]
            dim = K.shape(mu)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return mu + K.exp(log_var * 0.5) * epsilon
        return tf.keras.layers.Lambda(function=function)([mu, log_var])

    def conv2d(self, x, filters, kernel_size, strides, kernel_initializer='he_normal', bn=False, activation='relu'):
        x = tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            use_bias=False if bn else True,
            kernel_initializer=kernel_initializer)(x)
        if bn:
            x = tf.keras.layers.BatchNormalization()(x)
        return self.activation(x, activation)

    def conv2d_transpose(self, x, filters, kernel_size, strides, kernel_initializer='he_normal', bn=False, activation='relu'):
        x = tf.keras.layers.Conv2DTranspose(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            use_bias=False if bn else True,
            kernel_initializer=kernel_initializer)(x)
        if bn:
            x = tf.keras.layers.BatchNormalization()(x)
        return self.activation(x, activation)

    def dense(self, x, units, bn=False, activation='relu'):
        x = tf.keras.layers.Dense(
            units=units,
            use_bias=False if bn else True,
            kernel_initializer='he_normal')(x)
        if bn:
            x = tf.keras.layers.BatchNormalization()(x)
        return self.activation(x, activation)

    def activation(self, x, activation, alpha=0.1):
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

