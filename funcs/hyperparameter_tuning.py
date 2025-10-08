from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def best_svc_params_gridsearch(X_train, y_train, param_grid):
    """
    Performs a grid search to find the best hyperparameters for an SVC.
    """
    base_clf = SVC()
    grid_search = GridSearchCV(base_clf, param_grid=param_grid, cv=5, scoring='f1',verbose=3)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")


    #use grid_search.best_estimator_ as the best model to do .transform n all
    return grid_search

def best_svc_params_randomizedsearch(X_train, y_train, param_grid):
    """
    Performs randomized search to find the best hyperparameters for an SVC.
    """
    base_clf = SVC()
    rand_search = RandomizedSearchCV(base_clf,param_distributions=param_grid)
    rand_search.fit(X_train, y_train)

    print(f"Best parameters found: {rand_search.best_params_}")
    print(f"Best cross-validation score: {rand_search.best_score_:.4f}")

    return rand_search


