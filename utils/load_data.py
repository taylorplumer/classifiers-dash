import pandas as pd
import yellowbrick
from yellowbrick.datasets import load_credit
from sklearn.model_selection import train_test_split
from utils.upsample import upsample

def load_data(filepath, upsampled=False):

    df = pd.read_csv(filepath)
    target = df.columns[0]
    features = [col for col in df.columns if col not in target]
    X = df[features].values
    y = df[target].values
    labels = df[target].unique().tolist()
    labels.sort()

    train_df, test_df = train_test_split(df, test_size = .30, random_state=42)

    # ensure that upsample method only is applied to training set
    if upsampled==True:
        df_upsampled, X_train, y_train= upsample(train_df, target, features)
        X_test = test_df[features].values
        y_test = test_df[target].values

    else:
        X_train = train_df[features].values
        y_train = train_df[target].values
        X_test = test_df[features].values
        y_test = test_df[target].values


    return labels, features, target, X_train, X_test, y_train, y_test
