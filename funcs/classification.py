
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

#function to train and evaluate an svm classifier
#call this with your train and test data (x_train, y_train, x_test, y_test)
#supply additional svc parameters (kernel, c, etc.) via **kwargs if needed.
def classify_svc(x_train, y_train, x_test, y_test, **kwargs):
    #initialize svm model with any given hyperparameters
    clf = SVC(**kwargs)
  
    #fit the model on training set (features/labels)
    clf.fit(x_train, y_train)

    #predict class labels for the test set
    y_pred = clf.predict(x_test)

    #compute and print overall accuracy on test set
    accuracy_clf = accuracy_score(y_test, y_pred)
    print(f"classification accuracy: {accuracy_clf}")

    #print detailed precision/recall/f1 report per class
    print("classification report", classification_report(y_test, y_pred))

    #visualize the confusion matrix for test predictions
    cmd = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    cmd.plot(cmap=plt.cm.Blues)

#function for training and evaluating a random forest classifier
#use for multi-feature and multi-class problems. randomforestclassifier is robust to feature scaling.
def classify_rf(x_train, y_train, x_test, y_test, **kwargs):
    #initialize random forest model; reproducible results with random_state
    print(f"Hyperparameters passed: {kwargs}")
    rf_classifier = RandomForestClassifier(**kwargs, random_state=42)
    rf_classifier.fit(x_train, y_train)

    #generate predictions for the test set
    y_pred_clf = rf_classifier.predict(x_test)

    #accuracy score (proportion classified correctly)
    accuracy_clf = accuracy_score(y_test, y_pred_clf)
    print(f"classification accuracy: {accuracy_clf}")

    #per-class metrics
    print("classification report", classification_report(y_test, y_pred_clf))

    #confusion matrix visualization for error analysis
    conf_matrix = confusion_matrix(y_test, y_pred_clf)
    cmd = ConfusionMatrixDisplay(conf_matrix)
    cmd.plot(cmap=plt.cm.Blues)

#function for logistic regression classification (for binary or multiclass problems)
#handles linearly separable data; supports additional configuration via **kwargs.
def classify_logistic(x_train, y_train, x_test, y_test, **kwargs):
    #initialize the logistic regression model (increase max_iter if convergence warnings appear)
    clf = LogisticRegression(max_iter=1000, **kwargs)
    clf.fit(x_train, y_train)

    #predict target labels for test set
    y_pred = clf.predict(x_test)

    #calculate overall test accuracy
    accuracy_clf = accuracy_score(y_test, y_pred)
    print(f"classification accuracy: {accuracy_clf}")

    #print per-class statistics
    print("classification report", classification_report(y_test, y_pred))

    #plot the confusion matrix for true vs. predicted labels
    conf_matrix = confusion_matrix(y_test, y_pred)
    cmd = ConfusionMatrixDisplay(conf_matrix)
    cmd.plot(cmap=plt.cm.Blues)
