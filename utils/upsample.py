from sklearn.utils import resample
import pandas as pd

def upsample(df, target, features):

    # Up-sample Minority Class approach from Elite Data Science
    # https://elitedatascience.com/imbalanced-classes


    # Seperate majority and minority classes

    value_dict = dict(df[target].value_counts())
    majority_value = list({k: v for k, v in sorted(value_dict.items(), key=lambda item: item[1], reverse=True)}.keys())[0]

    df_majority = df[df[target] == majority_value]
    df_minority = df[df[target] != majority_value]

    majority_n_samples = df[target].value_counts()[0]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                replace=True,
                                n_samples=majority_n_samples,
                                random_state=42)

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    X_upsampled = df_upsampled[features].values
    y_upsampled = df_upsampled[target].values

    return df_upsampled, X_upsampled, y_upsampled
