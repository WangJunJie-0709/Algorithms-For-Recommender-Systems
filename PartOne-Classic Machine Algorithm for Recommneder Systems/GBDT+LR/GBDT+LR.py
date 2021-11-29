import pandas as pd
import numpy as np

def read_Data(path):
    Data = pd.read_csv(path, nrows=100000)
    Ratings = Data.pivot_table(index='user_id', columns='anime_id', values='rating')
    return Ratings

if __name__ == '__main__':
    path = '../Data/archive/rating_complete.csv'
    Ratings = read_Data(path)
    print(Ratings)