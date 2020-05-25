from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# modify depending on needs for sklearn classifiers and yellowbrick visualizers
MODELS = [GradientBoostingClassifier(), RandomForestClassifier(), LogisticRegression(max_iter=1000), GaussianNB() ]
#models = [GradientBoostingClassifier(), RandomForestClassifier()]
VISUALIZERS = ['ROCAUC','PrecisionRecallCurve', 'ClassificationReport','ConfusionMatrix']

INPUT_DATA_FILEPATH = 'Data/Input/'

OUTPUT_DATA_FILEPATH = 'Data/Output/'

IMG_OUTPUT_FILEPATH = 'Data/img/'
