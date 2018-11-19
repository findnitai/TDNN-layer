""" model implementation

    This is a simplified version that aims to test the TDNN layer.
"""

from keras.models import Sequential
from TDNN_layer import TDNNLayer
import numpy as np


class PeddintiModel:

    def __init__(self):
        self.model = None
        self.input_dim = 0

    """ create_model
        
    This function will create the model as defined in Peddinti's paper
    
    param:
        input_dim: input data dimension
    """
    def create_model(self, input_dim):
        self.input_dim = input_dim
        self.model = Sequential()
        self.model.add(TDNNLayer([-2, 2], sub_sampling=False, input_shape=(input_dim, 1)))
        self.model.add(TDNNLayer([-1, 2], sub_sampling=True))
        self.model.add(TDNNLayer([-3, 2], sub_sampling=True))
        self.model.add(TDNNLayer([-7, 2], sub_sampling=True, activation="softmax"))
        self.model.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=['accuracy'])
        self.model.summary()

    """ train_model
    
    This function will train the model.
    The training is performed in batch. This function can be called multiple time to train on more data.
    
    param:
        data: the data to train on
        truth: the truth table
    """
    def train_model(self):
        data = np.random.random((3200, self.input_dim, 1))
        truth = np.round(np.random.random((3200, self.input_dim - 21)))
        self.model.fit(data, truth, epochs=20)

    """ evaluate_model
    
    This function will return the model evaluation
    
    param:
        data: the data to evaluate
        truth: the truth table
    """
    def evaluate_model(self):
        data = np.random.random((3200, self.input_dim, 1))
        truth = np.round(np.random.random((3200, self.input_dim - 21)))
        return self.model.evaluate(data, truth)


if __name__ == "__main__":
    model = PeddintiModel()
    model.create_model(32)
    model.train_model()
    model.evaluate_model()
