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
from vae import VariationalAutoEncoder

if __name__ == '__main__':
    VariationalAutoEncoder(
        input_shape=(32, 32, 1),
        train_image_path=r'/train_data/mnist/train',
        validation_image_path=r'/train_data/mnist/validation',
        lr=0.001,
        batch_size=32,
        latent_dim=2,
        view_grid_size=15,
        iterations=10000,
        training_view=True).fit()

