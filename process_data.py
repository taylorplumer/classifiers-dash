import pandas as pd
import numpy as np
from itertools import combinations
from visualizers import Visualizer
from helpers import evaluate_model, save_report
from upsample import upsample
from load_data import load_data
from sklearn.model_selection import train_test_split
from pandas.io.json import json_normalize

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import resample


def create_report_df(upsampled=False):

    # modify depending on needs for sklearn classifiers and yellowbrick visualizers
    models = [GradientBoostingClassifier(), RandomForestClassifier(), LogisticRegression(), GaussianNB() ]
    visualizers = ['ClassificationReport', 'ROCAUC','PrecisionRecallCurve', 'ConfusionMatrix']

    df, labels, X, y = load_data()
    train_df, test_df = train_test_split(df, test_size = .30, random_state=42)

    # ensure that upsample method only is applied to training set
    if upsampled==True:
        df_upsampled, X_train, y_train= upsample(train_df, 'purchase', labels)
        X_test = test_df[labels].values
        y_test = test_df['purchase'].values

    else:
        X_train = train_df[labels].values
        y_train = train_df['purchase'].values
        X_test = test_df[labels].values
        y_test = test_df['purchase'].values

    # iterate through models and visualizers to create and save yellowbrick visualizers to img directory
    report_dict= {}
    for model_ in models:
        for visualizer_ in visualizers:
            viz = Visualizer(X, y, labels, model_, visualizer_, upsampled=upsampled)
            viz.evaluate()
            viz.save_img()

        # instantiate and fit model
        model = model_
        model.fit(X_train, y_train)
        # saves string value of model name as key and sklearn classification_report output_dict as value
        report_dict[str(model).split('(')[0]] = evaluate_model(model, X_test, y_test)

    # create pandas dataframe of report_dict and transpose
    report_df = pd.DataFrame.from_dict(report_dict).T

    # quick check to see whether report_df column structure is as expected
    if report_df.columns.tolist() == ['0', '1', 'accuracy', 'macro avg', 'weighted avg']:
        pass
    else:
        print("Warning: Column names aren't as expected. Verify report_df output_dict is correct.")
    report_df.columns = ['0', '1', 'accuracy', 'Macro Avg', 'Micro Avg' ]

    dict_columns = ['0', '1', 'Macro Avg', 'Micro Avg']
    keys = ['precision', 'recall', 'f1-score', 'support']

    # revise key values to personalize to its associated column i.e. from 'precision' to 'precision_0'
    def revise_dict(x, col, keys):
        new_keys = [key+'_'+col for key in keys]
        new_dict = dict(zip(new_keys, list(x.values())))
        return new_dict

    for col in dict_columns:
        report_df[col] = report_df[col].apply(lambda x: revise_dict(x, col, keys))

    # iterate row wise through dataframe to normalize dictionary values into flat tables
    for col in dict_columns:
        new_dict = {}
        for classifier in report_df.index.values.tolist():
            name = str(classifier) + '_df'
            new_dict[name]= json_normalize(report_df.loc[classifier][col])
            new_dict[name]['classifier'] = [classifier]

        # concat all classifier flat tables into one dataframe
        dict_df = pd.concat(list(new_dict.values())).reset_index().drop(columns=['index'], axis=1)

        # merge on existing report_df dataframe index and on dict_df 'classifier' column value
        report_df = report_df.merge(dict_df, how='left', left_index=True, left_on=None, right_on='classifier').set_index('classifier')

    # select only created columns
    report_df = report_df.iloc[:,5:]
    # sort columns and filter out 'support' related columns
    report_df = report_df[sorted([col for col in report_df.columns if 'support' not in col])]

    return report_df

report_df = create_report_df()
report_df.to_csv('Data/Output/report_df.csv')

report_df_upsampled = create_report_df(upsampled=True)
report_df_upsampled.to_csv('Data/Output/report_df_upsampled.csv')
