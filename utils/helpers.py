import ast
import pandas as pd
import numpy as np
import yellowbrick
import os
from pandas.io.json import json_normalize
import plotly.graph_objects as go
from sklearn.metrics import classification_report as classificationreport
from utils.visualizers import Visualizer



def create_img(X_train, X_test, y_train, y_test, labels, model, visualizer, upsampled,  IMG_OUTPUT_FILEPATH):
    viz = Visualizer(X_train, X_test, y_train, y_test, labels, model, visualizer, upsampled=upsampled)
    viz.evaluate()
    if upsampled == True:
        outpath_ = IMG_OUTPUT_FILEPATH +  str(model).split('(')[0] + '/' + visualizer + '_upsampled.png'
    else:
        outpath_ = IMG_OUTPUT_FILEPATH + str(model).split('(')[0] + '/' + visualizer + '.png'
    viz.visualizer.show(outpath=outpath_, clear_figure=True)



def evaluate_model(model, X_train, y_train, X_test, y_test):

    """
    Evaluates model by providing individual category and summary metrics of model performance
    Args:
        X_train
        y_train
        X_test: subset of X values withheld from the model building process
        Y_test: subset of Y values witheld from the model building process and used to evaluate model predictions

    Returns:
        report: classification report with evaluation metrics (f1, precision, recall, support)
    """
    model = model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classificationreport(y_test, y_pred, target_names= ["0", "1"], output_dict=True)

    return report


# revise key values to personalize to its associated column i.e. from 'precision' to 'precision_0'
def customize_dict_keys(x, col, keys):
    new_keys = [key+'_'+col for key in keys]
    new_dict = dict(zip(new_keys, list(x.values())))
    return new_dict

def normalize_to_flat(classifier, df, col):
    name = str(classifier) + '_df'
    new_df= json_normalize(df.loc[classifier][col])
    new_df['classifier'] = [classifier]
    return new_df


def revise_report_df(report_df):
    # quick check to see whether report_df column structure is as expected
    if report_df.columns.tolist() != ['0', '1', 'accuracy', 'macro avg', 'weighted avg']:
        print("Warning: Column names aren't as expected. Verify report_df output_dict is correct.")

    report_df.columns = ['0', '1', 'accuracy', 'Macro Avg', 'Micro Avg' ]

    dict_columns = ['0', '1', 'Macro Avg', 'Micro Avg']
    keys = ['precision', 'recall', 'f1-score', 'support']

    for col in dict_columns:
        # revise key values to personalize to its associated column i.e. from 'precision' to 'precision_0'
        report_df[col] = report_df[col].apply(lambda x: customize_dict_keys(x, col, keys))

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


def create_heatmap(df):

    """
    Create Plotly Heatmap graph object
    Args:

        df: transformed report_df pandas dataframe from the process_data.py file

    Returns:
        fig: Ploty Heatmap Figure that consists of data parameter and optional layout parameter

    """

    fig = go.Figure(data=go.Heatmap(
                       z=df.values.tolist(),
                       x=df.columns,
                       #y=[classifier for classifier in df.index.values.tolist()],
                        y = df.index.values.tolist(),
                       hoverongaps = False,
                        xgap = 3,
                        ygap = 3,
                        colorscale=[[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'], [0.2222222222222222, 'rgb(244,109,67)'], [0.3333333333333333, 'rgb(253,174,97)'], [0.4444444444444444, 'rgb(254,224,144)'], [0.5555555555555556, 'rgb(224,243,248)'], [0.6666666666666666, 'rgb(171,217,233)'], [0.7777777777777778, 'rgb(116,173,209)'], [0.8888888888888888, 'rgb(69,117,180)'], [1.0, 'rgb(49,54,149)']]
                        ),
                       )
    return fig
