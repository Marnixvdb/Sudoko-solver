import pandas as pd


class DataLoader:
    def __init__(self):
        pass

    def load_training_data(self):
        # Load csv data file into dataframe
        data = pd.read_csv("https://www.kaggle.com/bryanpark/sudoku/download", names=("quizzes", "solutions"))

        # Get number of features
        feature_count = data.shape[1] - 1

        # Split dataframe into X and y
        X, y = data.iloc[:, :feature_count], data.iloc[:, feature_count:]

        return X, y
