from visualizers import Visualizer
from helpers import create_img, evaluate_model, revise_dict, normalize_to_flat
from upsample import upsample
from load_data import load_data

import sys
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
from itertools import combinations

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import resample


def create_report_df(input_data_filepath, upsampled=False):

    # modify depending on needs for sklearn classifiers and yellowbrick visualizers
    models = [GradientBoostingClassifier(), RandomForestClassifier(), LogisticRegression(max_iter=1000), GaussianNB() ]
    #models = [GradientBoostingClassifier(), RandomForestClassifier()]
    visualizers = ['ClassificationReport', 'ROCAUC','PrecisionRecallCurve', 'ConfusionMatrix']

    #df, labels, X, y = load_data()
    df, labels, features, target, X, y = load_data(input_data_filepath)
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

    # iterate through models and visualizers to create and save yellowbrick visualizers to img directory
    img_results = [create_img(X, y, labels, model, visualizer, upsampled) for visualizer in visualizers for model in models]

    # saves string value of model name as key and sklearn classification_report output_dict as value
    report_dict = {str(model).split('(')[0]: evaluate_model(model, X_train, y_train, X_test, y_test) for model in models}

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


    for col in dict_columns:
        # revise key values to personalize to its associated column i.e. from 'precision' to 'precision_0'
        report_df[col] = report_df[col].apply(lambda x: revise_dict(x, col, keys))

        # iterate row wise through dataframe to normalize dictionary values into flat tables
        new_dict = {str(classifier) + '_df': normalize_to_flat(classifier, report_df, col) for classifier in report_df.index.values.tolist()}

        # concat all classifier flat tables into one dataframe
        dict_df = pd.concat(list(new_dict.values())).reset_index().drop(columns=['index'], axis=1)

        # merge on existing report_df dataframe index and on dict_df 'classifier' column value
        report_df = report_df.merge(dict_df, how='left', left_index=True, left_on=None, right_on='classifier').set_index('classifier')

    # select only created columns
    report_df = report_df.iloc[:,5:]
    # sort columns and filter out 'support' related columns
    report_df = report_df[sorted([col for col in report_df.columns if 'support' not in col])]

    return report_df


def main():

    if len(sys.argv) == 2:

        input_data_filepath = 'Data/Input/' + sys.argv[1]


        report_df = create_report_df(input_data_filepath)
        report_df.to_csv('Data/Output/report_df.csv')

        report_df_upsampled = create_report_df(input_data_filepath, upsampled=True)
        report_df_upsampled.to_csv('Data/Output/report_df_upsampled.csv')

    else:
        print('Please provide the filename of the data file in the Data/Input directory'\
              'containing the target and feature variables.')

if __name__ == '__main__':
    main()
