from config import *
from utils.visualizers import Visualizer
from utils.helpers import create_img, evaluate_model, customize_dict_keys, normalize_to_flat, revise_report_df
from utils.upsample import upsample
from utils.load_data import load_data

import sys
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
from itertools import combinations
import random

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import resample

def main():

    if len(sys.argv) == 2:
        input_data = INPUT_DATA_FILEPATH + sys.argv[1]
    else:
        print('Please provide the filename of the data file in the Data/Input directory'\
              'containing the target and feature variables.')


    for upsampled in [False, True]:

        # load data- refer to load_data.py script for train test split and how to structure input dataframe
        labels, features, target, X_train, X_test, y_train, y_test = load_data(input_data, upsampled=upsampled)

        # iterate through models and visualizers to create and save yellowbrick visualizers to img directory
        img_results = [create_img(X_train, X_test, y_train, y_test, labels, model, visualizer, upsampled, IMG_OUTPUT_FILEPATH) for visualizer in VISUALIZERS for model in MODELS]

        # saves string value of model name as key and sklearn classification_report output_dict as value
        report_dict = {str(model).split('(')[0]: evaluate_model(model, X_train, y_train, X_test, y_test) for model in MODELS}

        # create pandas dataframe of report_dict and transpose
        report_df = pd.DataFrame.from_dict(report_dict).T

        # format report_df dataframe for use in app.py Dash Plotly heatmap
        revised_report_df = revise_report_df(report_df)

        if upsampled ==True:
            revised_report_df.to_csv(OUTPUT_DATA_FILEPATH + 'report_df_upsampled.csv')
        else:
            revised_report_df.to_csv(OUTPUT_DATA_FILEPATH  + 'report_df.csv')


if __name__ == '__main__':
    main()
