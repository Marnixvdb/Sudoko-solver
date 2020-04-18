import pandas as pd


class DataLoader:
    def __init__(self):
        pass

    def load_training_data(self):
        # Load csv data file into dataframe
        data = pd.read_csv("https://storage.googleapis.com/kaggle-data-sets/595/1134/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1587469169&Signature=WyGIaYmGjLUlncSpx3s5Vsbd4WgNEHsV3JN9WAi2%2F13g9voezm%2FYo0rjR%2BndwtXFunQAlLdcJ612BbG1AtVp9OI36ZatKc9tnVc4C0fQczuzENp8IDtjWH9basvLNpMIoHxOxyksHxf9T9NHN%2FOovjqZaBbN1GGiM76p8Uicd%2FUYV1wQXBfGrmDoTAThkGEB2w6apl6nukUckESe1KpVw31g5HU3CmngYYhU8ulZZv5m8kxD7R2sYy21QWIZ698a13ztEwlhnsWVXg%2F67gxmlPDpiKyK5K27gltlUa5qzzd19mRRjCvD9qLpiOWT12OuUSBoJHQaQcKLXO3pB0oR2Q%3D%3D&response-content-disposition=attachment%3B+filename%3Dsudoku.zip", compression='zip')

        # Get number of features
        feature_count = data.shape[1] - 1

        # Split dataframe into X and y
        X, y = data.iloc[:, :feature_count], data.iloc[:, feature_count:]
        

        return X, y
