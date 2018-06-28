# -*- coding: utf-8 -*-

"""Implementing the TDNN layer in keras.

The code presented in this repo is an implementation of Peddinti1's paper
"A time delay neural network architecture for efficient modeling of long temporal contexts"

"""


from keras import backend as K
from keras.engine.base_layer import Layer
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
                 **kwargs):

        self.output_dim = output_dim
        self.input_context = input_context
        self.sub_sampling = sub_sampling
        super(TDNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(TDNNLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        return K.dot(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]-self.input_context[1]+self.input_context[0]
