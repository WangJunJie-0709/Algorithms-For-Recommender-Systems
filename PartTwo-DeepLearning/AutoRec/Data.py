import pandas as pd
from torch.utils.data import DataLoader, Dataset

def load_data(path):
    Data = pd.read_csv(path)
    return Data

class MyDataSet(object):
    def __init__(self):





if __name__ == '__main__':
