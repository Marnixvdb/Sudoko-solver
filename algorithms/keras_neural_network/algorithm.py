import sklearn # Always import sklearn to prevent conflicts that arrise when it is imported after Keras
from sklearn.preprocessing import LabelEncoder
from keras import metrics
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import pandas as pd


class Algorithm:
    def __init__(self, number_neurons_layer_1, number_neurons_layer_2):
        self.label_encoder = None
        self.model = None

        self.number_neurons_layer_1 = number_neurons_layer_1
        self.number_neurons_layer_2 = number_neurons_layer_2

    def fit(self, X, y):
        # Convert labels to integers
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)
        y = self.label_encoder.transform(y)

        # Split data
        mask = np.random.rand(len(X)) < 0.9
        X_train, X_early_stopping = X[mask], X[~mask]
        y_train, y_early_stopping = y[mask], y[~mask]

        # Model setup
        self.model = Sequential()
        self.model.add(Dense(self.number_neurons_layer_1, activation="relu", input_shape=X_train.shape[1:]))
        self.model.add(Dense(self.number_neurons_layer_2, activation="relu"))
        self.model.add(Dense(len(self.label_encoder.classes_), activation="softmax"))
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        # Fit model
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

    def predict(self, X):
        # Get probabilistic predictions
        y_pred = self.model.predict(X)

        # Create dataframe
        dataframe = pd.DataFrame(y_pred, columns=self.label_encoder.classes_)

        # Add class predictions to dataframe
        dataframe["class_prediction"] = self.label_encoder.classes_[np.argmax(y_pred, axis=1)]

        return dataframe
