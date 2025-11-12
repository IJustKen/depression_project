import numpy as np
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from matplotlib import pyplot as plt


def plot_result(y_true, y_pred, residuals, title="Comparison with true values"):
    plt.figure(figsize=(10, 8))
    # sns.scatterplot(x=y_pred, y=residuals, color='dodgerblue', edgecolor='k', s=60)

    # plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    plt.title(title, fontsize=14)
    plt.plot(np.arange(len(y_true)), y_true, color='blue', linewidth = 0.3)
    plt.plot(np.arange(len(y_pred)), y_pred, color='green', linewidth = 0.3)
    plt.xlabel("Data point", fontsize=12)
    plt.ylabel("CGPA", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def get_training_scores(X_train, y_train, X_test, y_test):
    scoreList = []

    for i in range(1, 5):
        poly = PolynomialFeatures(i)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        pred = model.predict(X_test_poly)

        scoreList.append(r2_score(y_test, pred))

    return scoreList


def regress_linear(X_train, y_train, X_test, y_test, degree=3):
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    pred = model.predict(X_test_poly)

    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)

    print(f"\\nPolynomial Linear Regression (Degree {degree})")
    print(f"R2 Score: {r2}")
    print(f"Mean Squared Error: {mse}")

    residuals = y_test - pred
    plot_result(y_test, pred, residuals, f"Poly Linear Reg D={degree}")


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
