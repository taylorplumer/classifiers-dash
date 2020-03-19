from sklearn.utils import resample
import pandas as pd

def upsample(df, column, labels):

    # Up-sample Minority Class approach from Elite Data Science
    # https://elitedatascience.com/imbalanced-classes


    # Seperate majority and minority classes
    df_majority = df[df[column] == 0]
    df_minority = df[df[column] == 1]

    majority_n_samples = df[column].value_counts()[0]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                replace=True,
                                n_samples=majority_n_samples,
                                random_state=42)

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    X_upsampled = df_upsampled[labels].values
    y_upsampled = df_upsampled[column].values

    return df_upsampled, X_upsampled, y_upsampled
