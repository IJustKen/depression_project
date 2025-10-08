import numpy as np
from src.plotting import plot_residual
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor


def get_training_scores(X, y):
    scoreList = []

    for i in range(1, 5):
        poly = PolynomialFeatures(i)
        X_poly = poly.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_poly, np.array(y), test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        scoreList.append(r2_score(y_test, pred))

    return scoreList


def regress_linear(X, y, degree=3):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, np.array(y), test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    
    print("\n--- Polynomial Linear Regression (Degree {}) ---".format(degree))
    print("R2 Score:", r2)
    print("Mean Squared Error:", mse)
    
    residuals = y_test - pred
    plot_residual(y_test, pred, residuals, f"Poly Linear Reg D={degree}")


def regress_bagging(X_train, y_train, X_test, y_test):
    regr = BaggingRegressor(estimator=SVR(), n_estimators=80, random_state=0, n_jobs=-1)
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Bagging Regressor (SVR Base) ---")
    print("R2 Score:", r2)
    print("Mean Squared Error:", mse)


def regress_rf(X_train, y_train, X_test, y_test):
    rf_regressor = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_regressor.fit(X_train, y_train)

    y_pred_reg = rf_regressor.predict(X_test)
    mse_reg = mean_squared_error(y_test, y_pred_reg)
    r2_reg = r2_score(y_test, y_pred_reg)

    print("\n--- Random Forest Regressor ---")
    print("R2 Score:", r2_reg)
    print("Mean Squared Error:", mse_reg)

    residuals = y_test - y_pred_reg
    return residuals, y_pred_reg


def compare_regs(X_train, y_train, X_test, y_test, regressors):
    print("\n--- Comparing Multiple Regression Models ---")
    for name, model in regressors.items():
        try:
            model.fit(X_train, y_train)
            y_pred_reg = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred_reg)
            r2 = r2_score(y_test, y_pred_reg)
            
            print(f"{name} Regressor - MSE: {mse:.4f}, R2 Score: {r2:.4f}")
            residuals = y_test - y_pred_reg

            plot_residual(y_test, y_pred_reg, residuals, name)

        except Exception as e:
            print(f"Error training/evaluating {name}: {e}")

