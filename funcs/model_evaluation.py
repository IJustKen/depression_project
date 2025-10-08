#function to compare multiple classifiers side-by-side
#supply a dictionary of sklearn model instances (e.g., {"svc": SVC(), "rf": RandomForestClassifier()})
#each classifier is fit, evaluated, and its confusion matrix is visualized.
def compare_clf(x_train, y_train, x_test, y_test, classifiers):
    for name, model in classifiers.items():
        model.fit(x_train, y_train)         #train the classifier
        y_pred_clf = model.predict(x_test)  #predict labels for the test set
        accuracy = accuracy_score(y_test, y_pred_clf)  #accuracy for this classifier
        print(f"{name} classifier - accuracy: {accuracy:.4f}")

        #visualize confusion matrices for each model to compare their predictions
        conf_matrix = confusion_matrix(y_test, y_pred_clf)
        ConfusionMatrixDisplay(conf_matrix).plot(cmap="Blues")
        plt.title(f"confusion matrix for {name} classifier")
        plt.show()
