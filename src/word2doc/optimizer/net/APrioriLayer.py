import keras
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class APrioriLayer(Layer):

    def __init__(self, aprioris, **kwargs):
        super(APrioriLayer, self).__init__(**kwargs)
        base = np.divide(1, len(aprioris)).astype('float32')
        self.aprioris = np.vectorize(lambda e: np.divide(e, base).astype('float32'))(aprioris)

    def call(self, inputs, training=None):
        def apriori_inputs():
            return keras.layers.Multiply()([inputs, self.aprioris])

        return K.in_train_phase(inputs, apriori_inputs, training=training)

    def get_config(self):
        config = {'aprioris': self.aprioris}
        base_config = super(APrioriLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape