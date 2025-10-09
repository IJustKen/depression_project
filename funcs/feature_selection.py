#input the df BEFORE scaling/normalizing
def feat_eng(df):
  # combining satisfaction from studies and job both and getting average
    df['Average_Satisfaction'] = (df['Study Satisfaction'] + df['Job Satisfaction']) / 2
  
  # this measures how resilient one is to pressure, we are assuming the hypothesis that if your CGPA is high
  # while having high pressure and studying a lot it means they are more resilient to depression.
  # we could assume the opposite for someone prone to depression.
    df['Resilience'] = df['CGPA'] * (df['Academic Pressure'] + df['Work Pressure']) * df['Work/Study Hours']
  
  # this feature measures the balance between stress and satisfaction. A high value might indicate an unhealthy balance.
    df['Pressure_Satisfaction_Ratio'] = df['Total_Pressure'] / (df['Average_Satisfaction'] + 0.01)
  # adding a small number to the denominator to avoid division by zero

    return df



import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def select_features_with_rf(X_train, y_train, threshold=0.01):
    """
    Selects features using Random Forest importance scores.
    
    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.
        threshold (float): The minimum importance score to keep a feature.
        
    Returns:
        list: A list of the names of the selected features.
    """
    # 1. Train the model
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # 2. Get feature importances
    importances = rf.feature_importances_
    feature_names = X_train.columns
    
    # Create a DataFrame for easy viewing
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("--- Feature Importances from Random Forest ---")
    print(feature_importance_df)
    
    # 3. Select features above the threshold
    selected_features = feature_importance_df[feature_importance_df['Importance'] > threshold]
    
    selected_feature_names = selected_features['Feature'].tolist()
    
    print(f"\\nSelected {len(selected_feature_names)} features out of {len(feature_names)} using a threshold of {threshold}.")
    print("Selected Features:", selected_feature_names)
    
    return selected_feature_names

# --- How to use it ---
# Assume X_train and y_train are ready (after encoding and engineering)
# selected_features_list = select_features_with_rf(X_train, y_train)

# Now you can create your final DataFrames with only the selected features
# X_train_final = X_train[selected_features_list]
# X_test_final = X_test[selected_features_list] # Make sure to apply to X_test as well!


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

def select_features_with_lasso(X_train, y_train):
    """
    Selects features using L1 (Lasso) regularization.
    
    Returns:
        list: A list of the names of the selected features.
    """
    # Note: For LASSO, it's important that your data is scaled first!
    # Make sure X_train is the output of a scaler.
    
    # C is the inverse of regularization strength; smaller C means stronger regularization
    # and fewer features selected.
    lasso_selector = SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
    )
    
    lasso_selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[lasso_selector.get_support()]
    
    print(f"\\n--- LASSO Selection Results ---")
    print(f"Selected {len(selected_features)} features.")
    print("Selected Features:", selected_features.tolist())
    
    return selected_features.tolist()

# --- How to use it ---
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
#
# selected_by_lasso = select_features_with_lasso(X_train_scaled_df, y_train)





