import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np


def num_cat_column_analysis(data:pd.DataFrame) -> tuple :
    for column in data.columns:
        print("=" * 60)
        print(f"Column Analysis for: {column}")
        unique_count = data[column].nunique()
        print(f"    - Total Unique Values (n): {unique_count}")
        is_numerical = pd.api.types.is_numeric_dtype(data[column])

        if is_numerical:
            print("    - Type: Numerical/High Cardinality")
            print(f"    - Min: {data[column].min()}")
            print(f"    - Max: {data[column].max()}")
            print(f"    - Mean: {data[column].mean():.2f}")
        else:
            print("    - Type: Categorical/Ordinal/Low Cardinality")
            print("    - Value Counts:")
            print(data[column].value_counts(dropna=False))

        print("=" * 60)
    numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    return (numerical_cols,categorical_cols)


def num_cat_column_analysis(data:pd.DataFrame) -> tuple :
    for column in data.columns:
        print("=" * 60)
        print(f"Column Analysis for: {column}")
        unique_count = data[column].nunique()
        print(f"    - Total Unique Values (n): {unique_count}")
        is_numerical = pd.api.types.is_numeric_dtype(data[column])

        if is_numerical:
            print("    - Type: Numerical/High Cardinality")
            print(f"    - Min: {data[column].min()}")
            print(f"    - Max: {data[column].max()}")
            print(f"    - Mean: {data[column].mean():.2f}")
        else:
            print("    - Type: Categorical/Ordinal/Low Cardinality")
            print("    - Value Counts:")
            print(data[column].value_counts(dropna=False))

        print("=" * 60)
    numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    return (numerical_cols,categorical_cols)


def rm_rows_with_val(data_cat,column_name,value_to_drop) -> pd.DataFrame :
    #Get the initial count of rows to see how many will be dropped
    initial_row_count = len(data_cat)
    rows_to_drop_count = (data_cat[column_name] == value_to_drop).sum()

    #Drop the rows
    #We use boolean indexing to select all rows EXCEPT those where the column equals 'Others'
    data_cleaned= data_cat[data_cat[column_name] != value_to_drop].copy()

    #Verify the operation
    final_row_count = len(data_cleaned)

    print("=" * 50)
    print(f"Dataset Size Before Drop: {initial_row_count} rows")
    print(f"Rows Dropped ({column_name} is '{value_to_drop}'): {rows_to_drop_count}")
    print(f"Dataset Size After Drop: {final_row_count} rows")
    print("=" * 50)

    #Update your main DataFrame reference
    return data_cleaned


def rm_rows_with_rare_cats(df: pd.DataFrame, min_count: int = 10) -> pd.DataFrame:
    initial_rows = len(df)
    # Create a boolean mask initialized to True (meaning keep all rows initially)
    # We will set a row to False (drop) if it contains a rare category
    mask = pd.Series(True, index=df.index)

    print(f"Starting with {initial_rows} rows.")
    print("-" * 40)

    for column in df.columns:
        is_numerical = pd.api.types.is_numeric_dtype(df[column])
        if is_numerical:
            continue
        else:
            #Calculate the frequency of each category in the current column
            category_counts = df[column].value_counts()

            #Identify the categories that are considered "rare"
            rare_categories = category_counts[category_counts < min_count].index

            #Create a temporary mask: True where the value is NOT a rare category
            #If the value is a rare category, the temporary mask is False
            temp_mask = ~df[column].isin(rare_categories)

            #Update the main mask: a row is kept only if ALL previous checks
            #AND the current check are True. If temp_mask is False (rare cat found),
            #the main mask is updated to False for that row.
            mask = mask & temp_mask

            print(f"| {column:<20}: {len(rare_categories)} rare categories found.")

    # Apply the final mask to the DataFrame
    df_cleaned = df[mask].copy()

    final_rows = len(df_cleaned)
    rows_removed = initial_rows - final_rows

    print("-" * 40)
    print(f"Processing Complete.")
    print(f"Total Rows Removed: {rows_removed}")
    print(f"Final Rows Remaining: {final_rows}")

    return df_cleaned


def one_hot_encode_dataframe(df_ohe: pd.DataFrame) -> pd.DataFrame:
    # Ensure all columns are treated as categorical/string before encoding
    for col in df_ohe.columns:
        df_ohe[col] = df_ohe[col].astype(str)

    ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')

    # 2. Use ColumnTransformer to apply the OHE to all columns
    # We create a simple passthrough transformer since ALL columns need OHE
    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', ohe_transformer, df_ohe.columns.tolist())
        ],
        remainder='passthrough', # This is just a formality here, but good practice
        verbose_feature_names_out=False # Use this for cleaner feature names (scikit-learn >= 1.2)
    )

    # 3. Fit and Transform the data (results in a NumPy array)
    data_encoded_array = preprocessor.fit_transform(df_ohe)

    # 4. Get the new feature names
    feature_names = preprocessor.get_feature_names_out()

    # 5. Convert the NumPy array back to a Pandas DataFrame
    df_encoded = pd.DataFrame(
        data_encoded_array,
        columns=feature_names,
        index=df_ohe.index
    )

    return df_encoded.rename(columns=lambda x: x.replace('ohe__', ''))


def Z_Scaler(data_num) -> pd.DataFrame :
    Scaler=StandardScaler()
    data_num_scaled_array=Scaler.fit_transform(data_num)
    data_num_scaled = pd.DataFrame(
        data_num_scaled_array,
        columns=data_num.columns,
        index=data_num.index
    )
    return data_num_scaled


def MinMax_Scaler(data_num) -> pd.DataFrame :
    Scaler=MinMaxScaler()
    data_num_scaled_array=Scaler.fit_transform(data_num)
    data_num_scaled = pd.DataFrame(
        data_num_scaled_array,
        columns=data_num.columns,
        index=data_num.index
    )
    return data_num_scaled
