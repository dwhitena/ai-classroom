from math import sqrt

import numpy as np
import pandas as pd

def squared_error(prediction, observation):
    """
    Calculates the squared error.

    Args:
        prediction - the prediction from our linear regression model
        observation - the observed data point
    Returns:
        The squared error
    """
    return (observation - prediction) ** 2

def sgd_fit(x, y, learning_rate, epochs):
    """
    Calculates the intercept and slope parameters using SGD for
    a linear regression model.

    Args:
        x - feature array
        y - response array
        learning_rate - learning rate
        epochs - the number of epochs to use in the SGD loop
    Returns:
        The intercept and slope parameters and the sum of
        squared error for the last epoch
    """

    # initialize the slope and intercept
    slope = 0.0
    intercept = 0.0

    # set the number of observations in the data
    N = float(len(y))

    # loop over the number of epochs
    for i in range(epochs):

        # calculate our current predictions
        predictions = (slope * x) + intercept
        
        # calculate the sum of squared errors for this epoch
        error = 0.0
        for idx, p in enumerate(predictions):
            error += squared_error(p, y[idx])
        error = error/N

        # calculate the gradients for the slope and intercept
        slope_gradient = -(2/N) * sum(x * (y - predictions))
        intercept_gradient = -(2/N) * sum(y - predictions)
        
        # update the slope and intercept
        slope = slope - (learning_rate * slope_gradient)
        intercept = intercept - (learning_rate * intercept_gradient)

    return intercept, slope, error

def ols_fit(x, y):
    """
    Calculates the intercept and slope parameters using OLS for
    a linear regression model.

    Args:
        x - feature array
        y - response array
    Returns:
        The intercept and slope parameters
    """
    
    # calculate the mean of x and y
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Using the derived OLS formula to calculate
    # the intercept and slope.
    numerator = 0
    denominator = 0
    for i in range(len(x)):
        numerator += (x[i] - mean_x) * (y[i] - mean_y)
        denominator += (x[i] - mean_x) ** 2
    slope = numerator / denominator
    intercept= mean_y - (slope * mean_x)

    return intercept, slope

def main():
    
    # import the data
    data = pd.read_csv('../data/training.csv')
    
    # fit our model using our various implementations
    int_sgd, slope_sgd, _ = sgd_fit(data['TV'].values, data['Sales'].values, 0.1, 1000)
    int_ols, slope_ols = ols_fit(data['TV'].values, data['Sales'].values)

    # output the results
    delim = "-----------------------------------------------------------------"
    print("\nOLS\n{delim}\n intercept: {intercept}, slope: {slope}"
            .format(delim=delim, intercept=int_ols, slope=slope_ols))
    print("\nSGD\n{delim}\n intercept: {intercept}, slope: {slope}"
            .format(delim=delim, intercept=int_sgd, slope=slope_sgd))
    print("")

if __name__ == "__main__":
    main()

