import pandas as pd
import yellowbrick
from yellowbrick.datasets import load_credit

def load_data():
    '''
    Load training data and return pandas dataframe, x and y
    '''

    df = pd.read_csv('Data/training.csv')

    labels = df.columns[3:].tolist()

    X = df[labels].values
    y = df['purchase'].values

    return df, labels, X, y

def load_credit_data():

    df = pd.read_csv('Data/credit.csv')
    features = [col for col in df.columns if col not in 'default']
    target = ['default']
    X = df[features].values
    y = df[target].values
    labels = df[target[0]].unique().tolist()
    labels.sort()


    return df, labels, features, target, X, y
