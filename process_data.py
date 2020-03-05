import pandas as pd
import numpy as np
from itertools import combinations
from classifiers import classification_report, rocauc, pr_curve, confusion_matrix
from evaluate import evaluate_model, save_report
from upsample import upsample
from sklearn.model_selection import train_test_split


from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import resample

#models = [GradientBoostingClassifier(), RandomForestClassifier()]
models = [GradientBoostingClassifier()]
#classifiers = [classification_report, rocauc, pr_curve, confusion_matrix]
classifiers = [rocauc]


df, labels, X, y = load_data()


df_upsampled, X_upsampled, y_upsampled = upsample(df, 'purchase')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)


report_dict = {}
for model_ in models:
    for i in range(len(classifiers)):
        classifiers[i](X, y, model_)
        classifiers[i](X_upsampled, y_upsampled, model_, upsampled=True)

    model = model_
    model.fit(X_train, y_train)
    report_dict[str(model).split('(')[0]] = evaluate_model(model, X_test, y_test)

report_df = pd.DataFrame.from_dict(report_dict)

report_df.to_csv('Data/Output/report_df.csv')
