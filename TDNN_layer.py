# -*- coding: utf-8 -*-

"""Implementing the TDNN layer in keras.

The code presented in this repo is an implementation of Peddinti1's paper
"A time delay neural network architecture for efficient modeling of long temporal contexts"

"""


from keras import backend as K
from keras.engine.base_layer import Layer
from keras import activations
import numpy as np


class TDNNLayer(Layer):
    """TDNNLayer

    TDNNLayer sounds like 1D conv with extra steps. Why not doing it with Keras ?

    This layer inherits the Layer class from Keras and is inspired by conv1D layer.

    The documentation will be added later.
    """

    def __init__(self,
                 input_context=[-2, 2],
                 sub_sampling=False,
                 initializer='uniform',
                 activation=None,
                 **kwargs):

        self.input_context = input_context
        self.sub_sampling = sub_sampling
        self.initializer = initializer
        self.activation = activations.get(activation)
        super(TDNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        kernel_shape = (self.input_context[1]-self.input_context[0]+1,1)
        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=self.initializer,
                                      trainable=True)
        self.mask = np.zeros(kernel_shape)
        self.mask[0][0] = 1
        self.mask[self.input_context[1]-self.input_context[0]][0] = 1

        if self.sub_sampling:
            self.kernel = self.kernel * self.mask

        super(TDNNLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        if self.sub_sampling:
            output = K.conv1D(inputs,
                              self.kernel,
                              stride=1,
                              padding=0,
                              )
        else:
            output = K.conv1D(inputs,
                              self.kernel * self.mask,
                              stride=1,
                              padding=0,
                              )
        if self.activation is not None:
            return self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]-self.input_context[1]+self.input_context[0]
