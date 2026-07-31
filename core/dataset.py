from config import *
from utils import *


class DatasetManager:
    def __init__(self):
        self.datasets = DATASETS

    def load(self):
        data = {}
        for name, path in self.datasets.items():
            print_title(f"Loading {name}")
            df = read_csv(path)
            df = convert_time(df)
            data[name] = df
            print(f"Samples : {len(df)}")
            print(f"Duration: {df['time_s'].iloc[-1]:.2f} s")
            print()
        return data

    def names(self):
        return list(self.datasets.keys())
