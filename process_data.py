import pandas as pd
import numpy as np
from itertools import combinations
from classifiers import classification_report, rocauc, pr_curve, confusion_matrix
from helpers import evaluate_model, save_report, clean_report_df
from upsample import upsample
from load_data import load_data
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import resample


def create_report_df(upsampled=False):

    models = [GradientBoostingClassifier(), RandomForestClassifier(), LogisticRegression(), GaussianNB() ]
    classifiers = [classification_report, rocauc, pr_curve, confusion_matrix]

    df, labels, X, y = load_data()
    train_df, test_df = train_test_split(df, test_size = .30, random_state=42)

    if upsampled==True:
        df_upsampled, X_train, y_train= upsample(train_df, 'purchase', labels)
        X_test = test_df[labels].values
        y_test = test_df['purchase'].values


    else:
        X_train = train_df[labels].values
        y_train = train_df['purchase'].values
        X_test = test_df[labels].values
        y_test = test_df['purchase'].values


    report_dict= {}
    for model_ in models:
        for i in range(len(classifiers)):
            classifiers[i](X, y, model_, upsampled=upsampled)

        model = model_
        model.fit(X_train, y_train)
        report_dict[str(model).split('(')[0]] = evaluate_model(model, X_test, y_test)

    report_df = clean_report_df(pd.DataFrame.from_dict(report_dict))

    return report_df

report_df = create_report_df()
report_df.to_csv('Data/Output/report_df.csv')

report_df_upsampled = create_report_df(upsampled=True)
report_df_upsampled.to_csv('Data/Output/report_df_upsampled.csv')
