import ast
import base64
import json

import pandas as pd
import numpy as np
from pandas.io.json import json_normalize

import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from itertools import combinations
from classifiers import classification_report, rocauc, pr_curve, confusion_matrix
from upsample import upsample
from load_data import load_data
from helpers import evaluate_model, save_report, clean_report_df, create_heatmap


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import resample


elements = ['Upsample', 'No Upsample']

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
            html.Img(id='rocauc-image', src='children', height=300),
            html.Img(id='pr_curve-image', src='children', height=300),
            html.Img(id='classification_report-image', src='children', height=300),
            html.Img(id='confusion_matrix-image', src='children', height=300),
            html.Div([
                html.Pre(id='hover-data', style={'paddingTop':35})
                ], style={'width':'30%'})
])

@app.callback(Output('heatmap-graph', 'figure'),
                [Input('sample-dropdown', 'value')])
def update_heatmap(sample_selection):

    if sample_selection == 'Upsample':
        filepath = 'Data/Output/report_df_upsampled.csv'
    else:
        filepath = 'Data/Output/report_df.csv'

    df = clean_report_df(filepath)
    figure = create_heatmap(df)

    return figure

@app.callback(
    [Output('rocauc-image', 'src'), Output('pr_curve-image', 'src'),
    Output('classification_report-image', 'src'), Output('confusion_matrix-image', 'src'),
    Output('hover-data', 'children')],
    [Input('sample-dropdown', 'value'), Input('heatmap-graph', 'hoverData')])
def callback_image(sample_selection, hoverData):
    path = '/Users/taylorplumer/Documents/2020/yellowbrick/Dash/'

    hover_dict = ast.literal_eval(json.dumps(hoverData, indent=2))

    model = hover_dict['points'][0]['y']


    vizs = ['rocauc', 'pr_curve', 'classification_report','confusion_matrix']
    #vizs = ['rocauc', 'pr_curve']
    image_dict = {}
    for viz in vizs:
        if sample_selection == 'Upsample':
            viz_path = 'Data/img/' + model + '/' + viz + '_upsampled.png'
            viz_image = encode_image(path+viz_path)
            image_dict[viz] = viz_image
        else:
            viz_path = 'Data/img/' + model + '/' + viz + '.png'
            viz_image = encode_image(path+viz_path)
            image_dict[viz] = viz_image


    return image_dict['rocauc'], image_dict['pr_curve'], image_dict['classification_report'], image_dict['confusion_matrix'], json.dumps(hoverData, indent=2)

if __name__ == '__main__':
    app.run_server()
