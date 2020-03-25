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
from visualizers import ClassificationReport, ROCAUC, PrecisionRecallCurve, ConfusionMatrix
from upsample import upsample
from load_data import load_data
from helpers import evaluate_model, save_report, clean_report_df, create_heatmap


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import resample


elements = ['Upsample', 'No Upsample']

report_df = pd.read_csv('Data/Output/report_df.csv').set_index('classifier')
report_df_upsampled = pd.read_csv('Data/Output/report_df_upsampled.csv').set_index('classifier')

app = dash.Dash()

def encode_image(image_file):
    encoded = base64.b64encode(open(image_file, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded.decode())


app.layout = html.Div([
            dcc.Dropdown(
                id='sample-dropdown',
                options=[{'label': element, 'value': element} for element in elements],
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
    Output('ClassificationReport-image', 'src'), Output('ConfusionMatrix-image', 'src'),
    Output('hover-data', 'children')],
    [Input('sample-dropdown', 'value'), Input('heatmap-graph', 'hoverData')])
def callback_image(sample_selection, hoverData):
    path = os.getcwd() + '/'
    hover_dict = ast.literal_eval(json.dumps(hoverData, indent=2))
    model_on_hover = hover_dict['points'][0]['y']
    visualizations = ['ClassificationReport', 'ROCAUC','PrecisionRecallCurve', 'ConfusionMatrix']

    image_dict = {}
    for visualization in visualizations:
        if sample_selection == 'Upsample':
            visualization_path = 'Data/img/' + model_on_hover + '/' + visualization + '_upsampled.png'
            visualization_image = encode_image(path+visualization_path)
            image_dict[visualization] = visualization_image
        else:
            visualization_path = 'Data/img/' + model_on_hover + '/' + visualization + '.png'
            visualization_image = encode_image(path+visualization_path)
            image_dict[visualization] = visualization_image

    return image_dict['ROCAUC'], image_dict['PrecisionRecallCurve'], image_dict['ClassificationReport'], image_dict['ConfusionMatrix'], json.dumps(hoverData, indent=2)

if __name__ == '__main__':
    app.run_server()
