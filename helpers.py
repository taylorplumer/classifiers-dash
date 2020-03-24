import ast
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import plotly.graph_objects as go
from sklearn.metrics import classification_report as classificationreport

def evaluate_model(model, X_test, Y_test):

    """
    Evaluates model by providing individual category and summary metrics of model performance
    Args:

        X_test: subset of X values withheld from the model building process
        Y_test: subset of Y values witheld from the model building process and used to evaluate model predictions

    Returns:
        report: classification report with evaluation metrics (f1, precision, recall, support)
    """
    y_pred = model.predict(X_test)

    report = classificationreport(y_pred, Y_test, target_names= ["0", "1"], output_dict=True)

    print(report)


    return report



def save_report(report, report_filepath='Data/Output/report.csv'):

    """
    Loads classification report to csv file
    Args:
        report: classification report returned from evaluate_model function
        report_filepath: path for where to save report
    Returns:
        report_df: save dataframe as a csv at specified file path
    """

    report_df = pd.DataFrame(report).transpose()

    report_df.columns = ['f1', 'precision', 'recall', 'support']

    #report_df['categories'] = report_df.index

    report_df = report_df[['f1', 'precision', 'recall', 'support']]

    report_df.to_csv(report_filepath)


    return report_df


def clean_report_df(df):
    report_df = df.T
    if report_df.columns.tolist() == ['0', '1', 'accuracy', 'macro avg', 'weighted avg']:
        pass
    else:
        return "Warning: Column names aren't as expected. Verify report_df output_dict is correct."
    report_df.columns = ['0', '1', 'accuracy', 'Macro Avg', 'Micro Avg' ]
    dict_columns = ['0', '1', 'Macro Avg', 'Micro Avg']
    keys = ['precision', 'recall', 'f1-score', 'support']


    def revise_dict(x, col, keys):
        new_keys = [key+'_'+col for key in keys]
        new_dict = dict(zip(new_keys, list(x.values())))
        return new_dict

    for col in dict_columns:
        report_df[col] = report_df[col].apply(lambda x: revise_dict(x, col, keys))

    for col in dict_columns:
        new_dict = {}
        for classifier in report_df.index.values.tolist():
            name = str(classifier) + '_df'
            new_dict[name]= json_normalize(report_df.loc[classifier][col])
            #display(new_dict[name])
            new_dict[name]['classifier'] = [classifier]
        dict_df = pd.concat(list(new_dict.values())).reset_index().drop(columns=['index'], axis=1)

        report_df = report_df.merge(dict_df, how='left', left_index=True, left_on=None, right_on='classifier').set_index('classifier')

    report_df = report_df.iloc[:,5:]
    report_df = report_df[sorted([col for col in report_df.columns if 'support' not in col])]

    return report_df


def create_heatmap(df):
    fig = go.Figure(data=go.Heatmap(
                       z=df.values.tolist(),
                       x=df.columns,
                       #y=[classifier for classifier in df.index.values.tolist()],
                        y = df.index.values.tolist(),
                       hoverongaps = False,
                        xgap = 3,
                        ygap = 3),
                       )
    return fig
