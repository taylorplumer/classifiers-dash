from config import *
import ast
import base64
import json
import os
import sys

import pandas as pd
import numpy as np
from pandas.io.json import json_normalize

import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from itertools import combinations
from yellowbrick.classifier import  ClassificationReport, ROCAUC, PrecisionRecallCurve, ConfusionMatrix

from utils.helpers import create_heatmap
from utils.upsample import upsample
from utils.load_data import load_data


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import resample

if len(sys.argv) == 2:
    run_id = sys.argv[1]

dropdown_values = ['Upsample', 'No Upsample']

# read both non-upsampled and upsampled report_df csv files as dataframes
report_df = pd.read_csv(OUTPUT_DATA_FILEPATH + 'report_df.csv').set_index('classifier')
report_df_upsampled = pd.read_csv(OUTPUT_DATA_FILEPATH + 'report_df_upsampled.csv').set_index('classifier')

# create Dash app
app = dash.Dash()

# helper function to render images
def encode_image(image_file):
    encoded = base64.b64encode(open(image_file, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded.decode())

header_ =     html.Div(
                    className="app-header",
                    children=[
                        html.Div('CLASSIFIER SELECTION TOOL', className="app-header--title")
                    ]
                )


app.layout = html.Div([
            header_,
            dcc.Dropdown(
                id='sample-dropdown',
                options=[{'label': dropdown_value, 'value': dropdown_value} for dropdown_value in dropdown_values],
                value='No Upsample'
            ),
            dcc.Graph(id='heatmap-graph'),
            html.Img(id='ROCAUC-image', src='children', height=300),
            html.Img(id='PrecisionRecallCurve-image', src='children', height=300),
            html.Img(id='ClassificationReport-image', src='children', height=300),
            html.Img(id='ConfusionMatrix-image', src='children', height=300),
            html.Div([
                html.Pre(id='hover-data', style={'paddingTop':35})
                ], style={'width':'30%'})
])

@app.callback(Output('heatmap-graph', 'figure'),
                [Input('sample-dropdown', 'value')])
def update_heatmap(sample_selection):
    if sample_selection == 'Upsample':
        df = report_df_upsampled
    else:
        df = report_df
    figure = create_heatmap(df)

    return figure

@app.callback(
    [Output('ROCAUC-image', 'src'), Output('PrecisionRecallCurve-image', 'src'),
    Output('ClassificationReport-image', 'src'), Output('ConfusionMatrix-image', 'src')],
    [Input('sample-dropdown', 'value'), Input('heatmap-graph', 'hoverData')])
def callback_image(sample_selection, hoverData):
    # set path to current working directory
    path = os.getcwd() + '/'

    # interprets hover data json as dictionary
    hover_dict = ast.literal_eval(json.dumps(hoverData, indent=2))
    model_on_hover = hover_dict['points'][0]['y']

    # create list of yellowbrick visualizers with naming conventions matching how they are stored in img diretory
    visualizations = VISUALIZERS

    # create dictionary with each value as an element of visualizations list and value as associated base65 image
    output_list = []
    for visualization in visualizations:
        if sample_selection == 'Upsample':
            visualization_path = IMG_OUTPUT_FILEPATH  + model_on_hover + '/' + visualization + '_upsampled.png'
            visualization_image = encode_image(path+visualization_path)
            output_list.append(visualization_image)
        else:
            visualization_path = IMG_OUTPUT_FILEPATH  + model_on_hover + '/' + visualization + '.png'
            visualization_image = encode_image(path+visualization_path)
            output_list.append(visualization_image)
    return output_list

if __name__ == '__main__':
    app.run_server()
