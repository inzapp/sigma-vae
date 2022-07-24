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
import cv2
import os
import numpy as np
import tensorflow as tf
from concurrent.futures.thread import ThreadPoolExecutor


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 image_paths,
                 input_shape,
                 batch_size,
                 latent_dim):
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.pool = ThreadPoolExecutor(8)
        self.img_index = 0

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        fs = []
        for _ in range(self.batch_size):
            fs.append(self.pool.submit(self.load_image, self.next_image_path()))
        batch_x = []
        for f in fs:
            img = f.result()
            img = self.resize(img, (self.input_shape[1], self.input_shape[0]))
            x = np.asarray(img).reshape(self.input_shape)
            batch_x.append(x)
        batch_x = self.normalize(np.asarray(batch_x).reshape((self.batch_size,) + self.input_shape).astype('float32'))
        return batch_x

    def next_image_path(self):
        path = self.image_paths[self.img_index]
        self.img_index += 1
        if self.img_index == len(self.image_paths):
            self.img_index = 0
            np.random.shuffle(self.image_paths)
        return path

    def load_image(self, image_path):
        return cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE if self.input_shape[-1] == 1 else cv2.IMREAD_COLOR)

    @staticmethod
    def normalize(x):
        return x / 255.0

    @staticmethod
    def denormalize(x):
        return x * 255.0

    @staticmethod
    def get_z_vector(size):
        return np.random.normal(loc=0.0, scale=1.0, size=size)

    @staticmethod
    def resize(img, size):
        if img.shape[1] > size[0] or img.shape[0] > size[1]:
            return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        else:
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

