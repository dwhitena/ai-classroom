import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def model(x):
    return 0.553*x + 0.215

def main():

    # import the data
    data = pd.read_csv('../data/training.csv')

    # calculate modeled values for Sales based on our 
    # linear regression model
    data['Model'] = data['TV'].apply(lambda x: model(x))

    # create a plot
    fig, ax = plt.subplots()
    ax.plot(data['TV'].values, data['Model'].values, 'k--')
    ax.plot(data['TV'].values, data['Sales'].values, 'ro')
    plt.xlabel('TV', axes=ax)
    plt.ylabel('Sales', axes=ax)

    # Only draw spine between the y-ticks
    ax.spines['left'].set_bounds(0, 1)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.show()

if __name__ == "__main__":
    main()
