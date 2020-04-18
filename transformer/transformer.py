import pandas as pd


class Transformer:
    def __init__(self):
        pass

    def get_data_row_by_id(self, identifier):
        # Dummy data
        return ["column 1 value", "column 2 value", "column 3 value"]

    def transform_input(self, inputs):
        row = self.get_data_row_by_id(inputs["id"])

        return pd.Series(row, index=["column 1 name", "column 2 name", "column 3 name"])
