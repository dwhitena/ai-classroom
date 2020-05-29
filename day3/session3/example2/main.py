import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def stat_parity(model, test_data, groups, labels, priv_group, unpriv_group, fav_result, unfav_result):
    """
    Calculate the Statistical Parity Difference for a classifier.

    Args:
        model - scikit learn classifier
        test_data - the test dataset in dataframe format
        groups - column in the df with the groups to be measured
        labels - column in the df with the results/labels
        priv_group - the priviledged group
        unpriv_group - the unpriviledged group
        fav_result - the favorable label/result
        unfav_result - the unfavorable label/result
    Returns:
        the difference in mean outcomes
    """

    # compute the rate of favorable results for the 
    # privileged and unprivileged groups
    predictions_priv = model.predict(test_data[test_data['sex'] == priv_group][['sex', 'education-num']])
    predictions_unpriv = model.predict(test_data[test_data['sex'] == unpriv_group][['sex', 'education-num']])
    priv_rate = predictions_priv.tolist().count(1)/len(predictions_priv)
    unpriv_rate = predictions_unpriv.tolist().count(1)/len(predictions_unpriv)

    return unpriv_rate - priv_rate 

def main():
    
    # dataset columns
    cols = [
      "age",
      "workclass",
      "fnlwgt",
      "education",
      "education-num",
      "marital-status",
      "occupation",
      "relationship",
      "race",
      "sex",
      "capital-gain",
      "capital-loss",
      "hours-per-week",
      "native-country",
      "income"
    ]
    
    # import dataset
    data = pd.read_csv("../data/income_data.csv", names=cols)

    # pre-process data
    data['sex'] = data['sex'].apply(lambda x: 1 if x == ' Male' else 0)
    data['income'] = data['income'].apply(lambda x: 1 if x == ' >50K' else 0)

    # split into training and test sets
    train, test = train_test_split(data, test_size=0.2)

    # train classifier
    model = LogisticRegression().fit(train[['sex', 'education-num']], train['income'])

    # calculate the Statistical Parity Difference for Male/Female
    stat_par = stat_parity(model, test, 'sex', 'income', 1, 0, 1, 0)
    print('Statistical Parity Difference: ', stat_par)

if __name__ == "__main__":
    main()
