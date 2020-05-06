from math import sqrt

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def main():
    
    # import the data
    data = pd.read_csv('../data/Advertising.csv')
    
    # scale the feature and response
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['TV', 'Sales']])

    # split the dataset into training and test sets
    data = pd.DataFrame(data_scaled, columns=['TV', 'Sales'])
    train = data.sample(frac=0.8,random_state=200) 
    test = data.drop(train.index)

    # export the data
    train.to_csv('../data/training.csv', index=False)
    test.to_csv('../data/test.csv', index=False)

if __name__ == "__main__":
    main()

