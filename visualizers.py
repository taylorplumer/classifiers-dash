from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ConfusionMatrix

class Visualizer():
    def __init__(self, X, y, labels, model, viz_selection, upsampled=False):
        self.X = X
        self.y = y
        self.labels = labels
        self.model = model
        self.viz_selection = viz_selection
        self.upsampled = upsampled
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = .30, random_state=42)

        if self.viz_selection == 'ClassificationReport':
            self.visualizer = ClassificationReport(self.model, classes=self.labels, support=True)
        elif self.viz_selection == 'ROCAUC':
            self.visualizer = ROCAUC(self.model, classes=self.labels, support=True)
        elif self.viz_selection == 'PrecisionRecallCurve':
            self.visualizer = PrecisionRecallCurve(self.model)
        elif self.viz_selection == 'ConfusionMatrix':
            self.visualizer = ConfusionMatrix(model, classes=self.labels)

    def evaluate(self):
        self.visualizer.fit(self.X_train, self.y_train)
        self.visualizer.score(self.X_test, self.y_test)

    def save_img(self):
        if self.upsampled == True:
            self.outpath_ = 'Data/img/' + str(self.model).split('(')[0] + '/' + self.viz_selection + '_upsampled.png'
        else:
            self.outpath_ = 'Data/img/' + str(self.model).split('(')[0] + '/' + self.viz_selection + '.png'
        return self.visualizer.show(outpath=self.outpath_, clear_figure=True)
