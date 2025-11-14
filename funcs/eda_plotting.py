import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#can input the entire DF in this
def analyze_feature_distributions(df, target_col='Depression'):
    """
    Analyzes and visualizes the distribution of each feature against a binary target variable.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing features and the target.
        target_col (str): The name of the binary target column.
    """
    
    #Separate feature columns from the target
    features = df.columns.drop(target_col)
    
    #Identify numerical and categorical features
    numerical_features = df[features].select_dtypes(include=['number']).columns
    categorical_features = df[features].select_dtypes(include=['object', 'category', 'bool']).columns
    
    print(f"Found {len(numerical_features)} numerical features.")
    print(f"Found {len(categorical_features)} categorical features.")

    # 1. Visualize Numerical Features 
    for feat in numerical_features:
        plt.figure(figsize=(10, 5))
        
        # Create a KDE plot for each class of the target variable
        sns.kdeplot(data=df, x=feat, hue=target_col, fill=True, common_norm=False, palette='viridis')
        
        plt.title(f'Distribution of "{feat}" by {target_col}', fontsize=14)
        plt.xlabel(feat, fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(title=target_col, labels=['Depression', 'No Depression'])
        plt.tight_layout()
        plt.show()

    #  2. Visualize Categorical Features 
    for feat in categorical_features:
        plt.figure(figsize=(12, 6))
        
        #create a count plot, grouped by the target variable
        sns.countplot(data=df, x=feat, hue=target_col, palette='viridis')
        
        plt.title(f'Distribution of "{feat}" by {target_col}', fontsize=14)
        plt.xlabel(feat, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right') #rotate labels for better readability
        plt.legend(title=target_col, labels=['No Depression', 'Depression'])
        plt.tight_layout()
        plt.show()


# analyze_feature_distributions(df, target_col='Depression')


# #One hot encoding the categorical features with the standard way of dropping first column
# df_encoded = pd.get_dummies(df, drop_first=True)


#Either one hot encode the categorical columns first, or only use the numerical columns while inputting the df here.
def plot_feature_target_correlations(df, target_col='Depression'):
    """
    Computes and plots the correlation of numerical features (or encoded categorical ones) with the target.
    
    Args:
        df (pd.DataFrame): DataFrame containing numerical features and the target.
        target_col (str): The name of the binary target column.
    """
    #Calculate correlation of all features with the target column
    # .corr() automatically ignores non-numeric columns
    correlation_series = df.corr()[target_col]
    
    #Remove the target's self-correlation (which is always 1)
    correlation_series = correlation_series.drop(target_col)
    
    #Sort the values in descending order for better visualization
    correlation_series = correlation_series.sort_values(ascending=False)
    
    #Plotting with Matplotlib
    plt.figure(figsize=(30, 8))
    
    #Create the bar plot
    bars = plt.bar(correlation_series.index, correlation_series.values, color=plt.cm.viridis(np.linspace(0, 1, len(correlation_series))))   
    #to give it the gradient effect
    
    plt.title(f'Correlation of Features with "{target_col}"', fontsize=16)
    plt.ylabel('Correlation Coefficient', fontsize=12)
    plt.xlabel('Features', fontsize=12)
    
    #Rotate feature names for readability
    plt.xticks(rotation=69, ha='right')
    
    #Add a horizontal line at y=0 for reference
    plt.axhline(0, color='grey', linewidth=1)
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# plot_feature_target_correlations(df_encoded, target_col='Depression')



#can input the entire df here it will automatically choose the numerical columns only within the function
def plot_grouped_boxplots(df, target_col='Depression'):
    """
    Generates a boxplot for each numerical feature, grouped by the target variable.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the binary target column.
    """
    
    # Select only the numerical columns, excluding the target itself
    numerical_features = df.select_dtypes(include=['number']).columns.drop(target_col, errors='ignore')
    
    print(f"Generating boxplots for {len(numerical_features)} numerical features...")
    
    for feature in numerical_features:
        plt.figure(figsize=(8, 5))
        
        # Create the boxplot using seaborn
        sns.boxplot(data=df, x=target_col, y=feature, palette='viridis')
        
        plt.title(f'Distribution of "{feature}" by {target_col}', fontsize=14)
        plt.xlabel(target_col, fontsize=12)
        plt.ylabel(feature, fontsize=12)
        plt.tight_layout()
        plt.show()

# plot_grouped_boxplots(df, target_col='Depression')

#For clustering
def plot_correlation_heatmap(df, figsize=(14, 10)):
    """
    Plots a correlation heatmap for all numerical features.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Select only numerical columns
    numeric_df = df.select_dtypes(include=['number'])

    plt.figure(figsize=figsize)
    corr = numeric_df.corr()

    sns.heatmap(corr, annot=False, cmap='viridis', center=0)
    
    plt.title("Correlation Heatmap (Numerical Features)", fontsize=16)
    plt.tight_layout()
    plt.show()

from sklearn.decomposition import PCA

def plot_pca_2d(df, labels=None, figsize=(8, 6)):
    """
    Reduces features to 2D using PCA and plots the projection.
    If labels are passed (e.g., cluster labels), points are colored accordingly.
    """
    import matplotlib.pyplot as plt

    # PCA only works on numeric features
    numeric_df = df.select_dtypes(include=['number'])

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(numeric_df)

    plt.figure(figsize=figsize)
    if labels is None:
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', alpha=0.7)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA 2D Projection")
    plt.tight_layout()
    plt.show()

    return reduced
