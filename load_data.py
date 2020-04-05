import pandas as pd
import yellowbrick
from yellowbrick.datasets import load_credit

def load_data(filepath):
    '''
    Load training data and return pandas dataframe, x and y
    '''

    df = pd.read_csv(filepath)
    target = df.columns[0]
    features = [col for col in df.columns if col not in target]
    X = df[features].values
    y = df[target].values
    labels = df[target].unique().tolist()
    labels.sort()


    return df, labels, features, target, X, y
