from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# modify depending on needs for sklearn classifiers and yellowbrick visualizers
MODELS = [GradientBoostingClassifier(), RandomForestClassifier(), LogisticRegression(max_iter=1000), GaussianNB() ]
VISUALIZERS = ['ROCAUC','PrecisionRecallCurve', 'ClassificationReport','ConfusionMatrix']

# modify depending on needs for filesystem structure
INPUT_DATA_FILEPATH = 'Data/Input/'

OUTPUT_DATA_FILEPATH = 'Data/Output/'

IMG_OUTPUT_FILEPATH = 'Data/img/'
