import pandas as pd

def mean_difference(df, groups, labels, priv_group, unpriv_group, fav_result, unfav_result):
    """
    Calculate the difference in mean outcomes between unprivileged 
    and privileged groups.

    Args:
        df - the dataset in dataframe format
        groups - column in the df with the groups to be measured
        labels - column in the df with the results/labels
        priv_group - the priviledged group
        unpriv_group - the unpriviledged group
        fav_result - the favorable label/result
        unfav_result - the unfavorable label/result
    Returns:
        the difference in mean outcomes
    """

    # compute the percentage of favorable results for the 
    # privileged and unprivileged groups
    priv_df = df[df[groups] == priv_group]
    unpriv_df = df[df[groups] == unpriv_group]
    per_priv = float(len(priv_df[priv_df[labels] == fav_result]))/float(len(priv_df))
    per_unpriv = float(len(unpriv_df[unpriv_df[labels] == fav_result]))/float(len(unpriv_df))

    return per_unpriv - per_priv

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

    # calculate the mean difference for Male/Female
    diff = mean_difference(data, 'sex', 'income', ' Male', ' Female', ' >50K', ' <=50K')
    print('Mean Difference: ', diff)

if __name__ == "__main__":
    main()
