import pandas as pd
from keras.datasets import mnist
import numpy as np

class DataLoader:
    def __init__(self):
        pass

    def load_training_data(self):
        # Load csv data file into dataframe
        (X_train,y_train),(X_test,y_test) = mnist.load_data()

        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test)) 

        return X, y
