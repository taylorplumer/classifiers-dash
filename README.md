# classifiers-dash-app
Build a Dash web app for binary classifier model selection

![](https://github.com/taylorplumer/classifiers-dash/blob/master/resources/classifier-dash-app_screenshot.png)

### Summary
This repository contains working code for deploying a binary classificaiton model selection tool to a Dash app locally. 

The web app primarily consists of two components:

1. A heatmap containing precision, recall, and f1 scores for each sklearn model along with their macro average (averaging the unweighted mean per label) and weighted average (averaging the support weighted mean per label).<sup>1</sup>
2. When hovering over associated sklearn model row in heatmap, images of matplotlib plots will populate that were created utilizing classification visualizers from the Yellowbrick project.<sup>2</sup>
    - ROCAUC: Graphs the receiver operating characteristics and area under the curve.
    - Precision-Recall Curves: Plots the precision and recall for different probability thresholds.
    - Classification Report: A visual classification report that displays precision, recall, and F1 per-class as a heatmap.
    - Confusion Matrix: A heatmap view of the confusion matrix of pairs of classes in multi-class classification.

A demo deployed to Heroku is available for viewing at the following address: <https://classifier-dash-app.herokuapp.com/>

The data used is the 'default of credit card clients Data Set' from the UCI Machine Learning Repository.<sup>3</sup>

### Instructions:
1. Input your data in the Data/Input directory with the target variable as the first column followed by the feature columns.
2. Run the following commands in the project's root directory to set up the data and model.

    - To create the yellowbrick classificaiton visualizer images and sklearn classificationreport output dictionaries (containing precision, recall, and f1 scores) for each sklearn model
        `python process_data.py`

3. Run the following command to run the Dash Plotly web app.
    `python app.py`

4. Go to http://127.0.0.1:8050/


###  Installation
After creating a virtual environment (recommended), you can install the dependencies with the following command: 

```
pip install -r requirements.txt
```

### References
<sup>1</sup> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

<sup>2</sup> https://www.scikit-yb.org/en/latest/api/classifier/index.html

<sup>3</sup> http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
