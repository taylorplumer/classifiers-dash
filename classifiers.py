from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ConfusionMatrix


def classification_report(X, y, model, upsampled=False):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)


    classes = ["No", "Yes"]

    visualizer = ClassificationReport(model, classes=classes, support=True)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)

    if upsampled == True:
        outpath_ = 'Data/img/' + str(model).split('(')[0] + '/classification_report_upsampled.png'
        viz_show = visualizer.show(outpath=outpath_, clear_figure=True)
    else:
        outpath_ = 'Data/img/' + str(model).split('(')[0] + '/classification_report.png'
        viz_show = visualizer.show(outpath=outpath_, clear_figure=True)

    return viz_show


def rocauc(X, y, model, upsampled=False):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)


    classes = ["No", "Yes"]

    visualizer = ROCAUC(model, classes=classes, support=True)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)

    if upsampled == True:
        outpath_ = 'Data/img/' + str(model).split('(')[0] + '/rocauc_upsampled.png'
        viz_show = visualizer.show(outpath=outpath_, clear_figure=True)
    else:
        outpath_ = 'Data/img/' + str(model).split('(')[0] + '/rocauc.png'
        viz_show = visualizer.show(outpath=outpath_, clear_figure=True)

    return viz_show


def pr_curve(X, y, model, upsampled=False):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)


    classes = ["No", "Yes"]

    visualizer = PrecisionRecallCurve(model)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)

    if upsampled == True:
        outpath_ = 'Data/img/' + str(model).split('(')[0] + '/pr_curve_upsampled.png'
        viz_show = visualizer.show(outpath=outpath_, clear_figure=True)
    else:
        outpath_ = 'Data/img/' + str(model).split('(')[0] + '/pr_curve.png'
        viz_show = visualizer.show(outpath=outpath_, clear_figure=True)

    return viz_show


def confusion_matrix(X, y, model, upsampled=False):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)

    classes = ["No", "Yes"]

    visualizer = ConfusionMatrix(model, classes=classes)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)

    if upsampled == True:
        outpath_ = 'Data/img/' + str(model).split('(')[0] + '/confusion_matrix_upsampled.png'
        viz_show = visualizer.show(outpath=outpath_, clear_figure=True)
    else:
        outpath_ = 'Data/img/' + str(model).split('(')[0] + '/confusion_matrix.png'
        viz_show = visualizer.show(outpath=outpath_, clear_figure=True)

    return viz_show
