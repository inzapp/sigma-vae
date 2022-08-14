"""
Authors : inzapp

Github url : https://github.com/inzapp/sigma-vae

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
import numpy as np


class LRScheduler:
    def __init__(self,
                 iterations,
                 lr=0.001,
                 min_lr=0.0,
                 min_momentum=0.85,
                 max_momentum=0.95,
                 initial_cycle_length=2500,
                 cycle_weight=2):
        self.lr = lr
        self.min_lr = min_lr
        self.max_lr = self.lr
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum
        self.iterations = iterations
        self.cycle_length = initial_cycle_length
        self.cycle_weight = cycle_weight
        self.cycle_step = 0

    def __set_lr(self, optimizer, lr):
        optimizer.__setattr__('lr', lr)

    def __set_momentum(self, optimizer, momentum):
        attr = ''
        if optimizer.__str__().lower().find('sgd') > -1:
            attr = 'momentum'
        elif optimizer.__str__().lower().find('adam') > -1:
            attr = 'beta_1'
        if attr != '':
            optimizer.__setattr__(attr, momentum)
        else:
            print(f'__set_momentum() failure. sgd and adam is available optimizers only.')

    def schedule_step_decay(self, optimizer, iteration_count, burn_in=1000):
        if iteration_count <= burn_in:
            lr = self.lr * pow(iteration_count / float(burn_in), 4)
        elif iteration_count == int(self.iterations * 0.5):
            lr = self.lr * 0.1
        elif iteration_count == int(self.iterations * 0.75):
            lr = self.lr * 0.01
        else:
            lr = self.lr
        self.__set_lr(optimizer, lr)
        return lr

    def schedule_one_cycle(self, optimizer, iteration_count):
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + np.cos(((1.0 / (self.iterations * 0.5)) * np.pi * iteration_count) + np.pi))  # up and down
        self.__set_lr(optimizer, lr)
        momentum = self.min_momentum + 0.5 * (self.max_momentum - self.min_momentum) * (1.0 + np.cos(((1.0 / (self.iterations * 0.5)) * np.pi * (iteration_count % self.iterations))))  # down and up
        self.__set_momentum(optimizer, momentum)
        return lr

    def schedule_cosine_warm_restart(self, optimizer, iteration_count, burn_in=1000):
        if iteration_count <= burn_in:
            lr = self.lr * pow(iteration_count / float(burn_in), 4)
        else:
            if self.cycle_step % self.cycle_length == 0 and self.cycle_step != 0:
                self.cycle_step = 0
                self.cycle_length = int(self.cycle_length * self.cycle_weight)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + np.cos(((1.0 / self.cycle_length) * np.pi * (self.cycle_step % self.cycle_length))))  # down and down
            self.cycle_step += 1
        self.__set_lr(optimizer, lr)
        return lr

