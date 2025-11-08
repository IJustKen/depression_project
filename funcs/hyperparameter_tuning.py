from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV#,HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv

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

def best_rf_params_gridsearch(X_train, y_train, param_grid):
    rf = RandomForestClassifier(random_state=42)

    # Set up GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

    # Fit GridSearchCV to your data
    grid_search.fit(X_train, y_train)
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    return grid_search

# def best_rf_params_halvinggridsearch(X_train, y_train, param_grid):
#     halving_search = HalvingGridSearchCV(
#         estimator = RandomForestClassifier(random_state=42),
#         param_grid = param_grid,
#         factor = 2,
#         cv = 5,
#         n_jobs=-1)
#     halving_search.fit(X_train, y_train)
#     print(f"Best parameters found: {halving_search.best_params_}")
#     print(f"Best CV score: {halving_search.best_score_}")
    
#     return halving_search

def best_rf_params_randomizedsearch(X_train, y_train, param_grid):
    rand_search = RandomizedSearchCV(
        estimator= RandomForestClassifier(),
        param_distributions = param_grid,
        n_iter = 300,
        cv = 5,
        n_jobs = -1,
        random_state = 42)
    rand_search.fit(X_train, y_train)
    print(f"Best parameters found: {rand_search.best_params_}")
    print(f"Best score: {rand_search.best_score_}")

    return rand_search


