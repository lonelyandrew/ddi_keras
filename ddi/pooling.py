#!/usr/bin/env python3

from keras import backend as K
from keras.engine.topology import Layer


class AttentionPooling(Layer):

    def __init__(self, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(shape=(input_shape[2], input_shape[2]),
                                 initializer='glorot_uniform', name='W',
                                 trainable=True)
        self.b = self.add_weight(shape=(1, input_shape[2]),
                                 initializer='glorot_uniform', name='b',
                                 trainable=True)
        self.uw = self.add_weight(shape=(input_shape[2], 1),
                                 initializer='glorot_uniform', name='b',
                                 trainable=True)

        super(AttentionPooling, self).build(input_shape)

    def call(self, inputs):
        U = K.map_fn(lambda x: K.dot(x, self.W), inputs)
        U = U + self.b
        U = K.tanh(U)
        U = K.map_fn(lambda x: K.dot(x, self.uw), U)
        U = K.permute_dimensions(U, (0, 2, 1))
        U = K.softmax(U)
        U = K.batch_dot(U, inputs)
        U = K.squeeze(U, 1)
        return U

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


if __name__ == '__main__':
    pass
