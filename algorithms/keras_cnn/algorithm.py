import numpy as np
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Lambda, MaxPooling2D # convolution layers
from tensorflow.keras.layers import Dense, Flatten # core layers
from keras.callbacks import EarlyStopping

class Algorithm:
    def __init__(self, number_neurons_layer_1):
        self.number_neurons_layer_1 = number_neurons_layer_1
        self.model = None

    def predict(self, X):
        X = X.reshape(28, 28, 1)
        y = self.model.prdict(X)
        return y
       
    def fit(self, X, y):
        X = X.reshape(60000, 28, 28, 1) / 255

        # Split data
        mask = np.random.rand(len(X)) < 0.9
        X_train, X_early_stopping = X[mask], X[~mask]
        y_train, y_early_stopping = y[mask], y[~mask]

        y = np_utils.to_categorical(y)
        num_classes=y.shape[1]

        ## Declare the model
        self.model = Sequential()

        ## Declare the layers
        layer_1 = Conv2D(32, kernel_size=3, activation='relu', input_shape=X_train.shape[1:])
        layer_2 = Conv2D(64, kernel_size=3, activation='relu')
        layer_3 = Flatten()
        layer_4 = Dense(10, activation='softmax')

        ## Add the layers to the model
        self.model.add(layer_1)
        self.model.add(layer_2)
        self.model.add(layer_3)
        self.model.add(layer_4)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)

        early_stopping = EarlyStopping(monitor="val_loss", mode="min")
        self.model.fit(
            X_train,
            y_train,
            batch_size=128,
            epochs=500,
            shuffle=True,
            verbose=1,
            validation_data=(X_early_stopping, y_early_stopping),
            callbacks=[early_stopping],
        )
