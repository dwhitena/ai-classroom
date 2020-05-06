import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def model(x):
    return 0.553*x + 0.215

def main():

    # import the test data
    data = pd.read_csv('../data/test.csv')

    # calculate modeled values for Sales based on our 
    # linear regression model
    data['Model'] = data['TV'].apply(lambda x: model(x))

    # calculate our RMSE
    rmse = math.sqrt(mean_squared_error(data['Model'].values, data['Sales'].values))
    print('RMSE: %0.4f'% rmse)

if __name__ == "__main__":
    main()
