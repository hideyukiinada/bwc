#!/usr/bin/env python
'''Set up CNN using Keras framework.

Credit
------
    I am using a portion of the code from the CNN implementation in the following Keras example files.
    Please see the license at the end of this docstring for their license terms.

    keras/examples/mnist_denoising_autoencoder.py
    keras/examples/mnist_cnn.py

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT license"
__email__ = "hideyuki@gmail.com"

=========================================================================================
License and copyright statements for the portion of the source code taken from the Keras
mnist_cnn.py examples code
=========================================================================================
COPYRIGHT

All contributions by François Chollet:
Copyright (c) 2015 - 2018, François Chollet.
All rights reserved.

All contributions by Google:
Copyright (c) 2015 - 2018, Google, Inc.
All rights reserved.

All contributions by Microsoft:
Copyright (c) 2017 - 2018, Microsoft, Inc.
All rights reserved.

All other contributions:
Copyright (c) 2015 - 2018, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.

LICENSE

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=========================================================================================
End of Keras license
'''

import os
import logging

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Convolution2DTranspose
from keras.layers import LeakyReLU
from keras.layers import MaxPooling2D
from keras.layers import Reshape



from project.config import load_config

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))  # Change the 2nd arg to INFO to suppress debug logging
config = load_config()


def cnn():
    """Convolutional Neural Network implementation

    Returns
    -------
    model: Reference to Keras Sequaltial model
        Set up and compile CNN to train the model to generate a color image from a grayscale image.
    """

    input_shape = (config["image_height"], config["image_width"], config["number_of_channels_in_source_image"])

    # Changes made due to the description in the DCGAN paper:
    #   No MaxPooling layer. Use Conv2D with strides=2
    #   Use LeakyReLU with alpha=0.2 inread of ReLU
    #   Adam learning rate changed to 0.0002 from lr=0.001
    #   Adam Beta1 to 0.5 from 0.9
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), padding='same', strides=2, input_shape=input_shape)) # to 128x128
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(64, kernel_size=(5, 5), padding='same', strides=2)) # to 64x64
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', strides=2)) # to 32x32
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', strides=2)) # to 16x16
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', strides=2)) # to 8x8
    model.add(LeakyReLU(alpha=0.2))

    #model.add(Conv2D(1024, kernel_size=(3, 3), padding='same', strides=2)) # to 4x4
    #model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
#    model.add(Dense(4096, activation='relu')) # bottleneck. Original image 196608 pixels (65536 pixels * 3 channels)
    model.add(Dense(2048, activation='relu')) # bottleneck. Original image 65536 pixels.
    model.add(Dense(16384*2, activation='sigmoid'))

    model.add(Reshape((8, 8, 512))) # 8*8*512 = 33K

    #model.add(Convolution2DTranspose(filters=512,
    #                        kernel_size=(3, 3),
    #                        strides=2,
    #                        padding='same')) # to 8x8
    #model.add(LeakyReLU(alpha=0.2))

    model.add(Convolution2DTranspose(filters=256,
                            kernel_size=(3, 3),
                            strides=2,
                            padding='same')) # to 16x16
    model.add(LeakyReLU(alpha=0.2))

    model.add(Convolution2DTranspose(filters=128,
                            kernel_size=(3, 3),
                            strides=2,
                            padding='same')) # to 32x32
    model.add(LeakyReLU(alpha=0.2))

    model.add(Convolution2DTranspose(filters=64,
                            kernel_size=(3, 3),
                            strides=2,
                            padding='same')) # to 64x64
    model.add(LeakyReLU(alpha=0.2))

    model.add(Convolution2DTranspose(filters=32,
                            kernel_size=(5, 5),
                            strides=2,
                            padding='same')) # to 128x128
    model.add(LeakyReLU(alpha=0.2))

    model.add(Convolution2DTranspose(filters=3,
                            kernel_size=(7, 7),
                            strides=2,
                            activation='sigmoid',
                            padding='same')) # to 256x256

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                  metrics=['accuracy'])

    return model
