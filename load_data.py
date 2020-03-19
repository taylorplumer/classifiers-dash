import pandas as pd

def load_data():
    '''
    Load training data and return pandas dataframe, x and y
    '''

    df = pd.read_csv('Data/training.csv')

    labels = df.columns[3:].tolist()

    X = df[labels].values
    y = df['purchase'].values

    return df, labels, X, y
