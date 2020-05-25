from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ConfusionMatrix


class Visualizer():
    def __init__(self, X_train, X_test, y_train, y_test, labels, model, viz_selection, upsampled=False):
        self.labels = labels
        self.model = model
        self.viz_selection = viz_selection
        self.upsampled = upsampled
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        if self.viz_selection == 'ClassificationReport':
            self.visualizer = ClassificationReport(self.model, classes=self.labels, support=True)
        elif self.viz_selection == 'ROCAUC':
            self.visualizer = ROCAUC(self.model, classes=self.labels, support=True)
        elif self.viz_selection == 'PrecisionRecallCurve':
            self.visualizer = PrecisionRecallCurve(self.model)
        elif self.viz_selection == 'ConfusionMatrix':
            self.visualizer = ConfusionMatrix(model, classes=self.labels)
        else:
            return print("Error: viz_selection does not match accepted values. View Visualizer Class for accepted values.")




    def evaluate(self):
        self.visualizer.fit(self.X_train, self.y_train)
        self.visualizer.score(self.X_test, self.y_test)

    def save_img(self):
        if self.upsampled == True:
            self.outpath_ = IMG_OUTPUT_FILEPATH + str(self.model).split('(')[0] + '/' + self.viz_selection + '_upsampled.png'
        else:
            self.outpath_ = IMG_OUTPUT_FILEPATH + str(self.model).split('(')[0] + '/' + self.viz_selection + '.png'
        return self.visualizer.show(outpath=self.outpath_, clear_figure=True)
