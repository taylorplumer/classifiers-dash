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

    report = classificationreport(y_pred, Y_test, target_names= ["No", "Yes"], output_dict=True)

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
