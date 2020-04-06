# classifiers-dash-app
Build a Dash web app diagnostic tool for binary classifier model selection

### Summary
The repository contains working code for deploying to a Dash app locally. Instructions are below.

A demo deployed to Heroku is available for viewing at the following address: <https://classifier-dash-app.herokuapp.com/>

The data used is the 'default of credit card clients Data Set' from the UCI Machine Learning Repository.

### Instructions:
1. Input your data in the Data/Input directory with the target variable as the first column followed by the feature data.
2. Run the following commands in the project's root directory to set up the data and model.

    - To create yellowbrick visualizers images and report dataframe consisting of sklearn classificationreport output dictionary for each sklearn model
        `python process_data.py`

3. Run the following command to run the Dash Plotly web app.
    `python app.py`

4. Go to http://127.0.0.1:8050/


###  Installation
After creating a virtual environment (recommended), you can install the dependencies with the following command: 

```
pip install -r requirements.txt
```
