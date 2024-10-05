from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_cover_samples():
  
    # fetch dataset
    covertype = fetch_ucirepo(id=31)


    # data (as pandas dataframes)

    X = covertype.data.features

    y = covertype.data.targets

    # metadata

    print(covertype.metadata)

    # variable information

    print(covertype.variables)


    # standardize X

    scaler = StandardScaler()

    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # write standardized X to a CSV file without headers

    X_scaled.to_csv('cover_type_x.csv', index=False, header=False)

    # transform y

    mid_value = (y.iloc[:, 0].max() + y.iloc[:, 0].min()) / 2

    y_transformed = y.iloc[:, 0].map(lambda x: -1 if x < mid_value else 1)

    # write transformed y to a CSV file without headers

    y_transformed.to_csv('cover_type_y.csv', index=False, header=False)


def generate_cover_10000_samples():
        
    # fetch dataset
    covertype = fetch_ucirepo(id=31)

    # data (as pandas dataframes)
    X = covertype.data.features
    y = covertype.data.targets

    # Extract 10000 samples
    X = X.sample(n=10000, random_state=1)
    y = y.loc[X.index]

    # metadata
    print(covertype.metadata)

    # variable information
    print(covertype.variables)

    # standardize X
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Check for missing values in X_scaled
    if X_scaled.isnull().values.any():
        print("Warning: X_scaled contains missing values.")
        X_scaled = X_scaled.dropna()

    # Check that all rows have the same number of columns
    num_cols = len(X_scaled.columns)
    for i, row in X_scaled.iterrows():
        if len(row) != num_cols:
            print(f"Warning: Row {i} has {len(row)} columns instead of {num_cols}")

    # write standardized X to a CSV file without headers
    X_scaled.to_csv('cover_type_x_10000.csv', index=False, header=False)

    # transform y
    mid_value = (y.iloc[:, 0].max() + y.iloc[:, 0].min()) / 2
    y_transformed = y.iloc[:, 0].map(lambda x: -1 if x < mid_value else 1)

    # Check for missing values in y_transformed
    if y_transformed.isnull().values.any():
        print("Warning: y_transformed contains missing values.")
        y_transformed = y_transformed.dropna()

    # write transformed y to a CSV file without headers
    y_transformed.to_csv('cover_type_y_10000.csv', index=False, header=False)


def generate_cover_50000_samples():
        
    # fetch dataset
    covertype = fetch_ucirepo(id=31)

    # data (as pandas dataframes)
    X = covertype.data.features
    y = covertype.data.targets

    # Extract 50000 samples
    X = X.sample(n=50000, random_state=1)
    y = y.loc[X.index]

    # metadata
    print(covertype.metadata)

    # variable information
    print(covertype.variables)

    # standardize X
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Check for missing values in X_scaled
    if X_scaled.isnull().values.any():
        print("Warning: X_scaled contains missing values.")
        X_scaled = X_scaled.dropna()

    # Check that all rows have the same number of columns
    num_cols = len(X_scaled.columns)
    for i, row in X_scaled.iterrows():
        if len(row) != num_cols:
            print(f"Warning: Row {i} has {len(row)} columns instead of {num_cols}")

    # write standardized X to a CSV file without headers
    X_scaled.to_csv('cover_type_x_50000.csv', index=False, header=False)

    # transform y
    mid_value = (y.iloc[:, 0].max() + y.iloc[:, 0].min()) / 2
    y_transformed = y.iloc[:, 0].map(lambda x: -1 if x < mid_value else 1)

    # Check for missing values in y_transformed
    if y_transformed.isnull().values.any():
        print("Warning: y_transformed contains missing values.")
        y_transformed = y_transformed.dropna()

    # write transformed y to a CSV file without headers
    y_transformed.to_csv('cover_type_y_50000.csv', index=False, header=False)


def generate_cover_100000_samples():
        
    # fetch dataset
    covertype = fetch_ucirepo(id=31)

    # data (as pandas dataframes)
    X = covertype.data.features
    y = covertype.data.targets

    # Extract 100000 samples
    X = X.sample(n=100000, random_state=1)
    y = y.loc[X.index]

    # metadata
    print(covertype.metadata)

    # variable information
    print(covertype.variables)

    # standardize X
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Check for missing values in X_scaled
    if X_scaled.isnull().values.any():
        print("Warning: X_scaled contains missing values.")
        X_scaled = X_scaled.dropna()

    # Check that all rows have the same number of columns
    num_cols = len(X_scaled.columns)
    for i, row in X_scaled.iterrows():
        if len(row) != num_cols:
            print(f"Warning: Row {i} has {len(row)} columns instead of {num_cols}")

    # write standardized X to a CSV file without headers
    X_scaled.to_csv('cover_type_x_100000.csv', index=False, header=False)

    # transform y
    mid_value = (y.iloc[:, 0].max() + y.iloc[:, 0].min()) / 2
    y_transformed = y.iloc[:, 0].map(lambda x: -1 if x < mid_value else 1)

    # Check for missing values in y_transformed
    if y_transformed.isnull().values.any():
        print("Warning: y_transformed contains missing values.")
        y_transformed = y_transformed.dropna()

    # write transformed y to a CSV file without headers
    y_transformed.to_csv('cover_type_y_100000.csv', index=False, header=False)

def generate_cover_200000_samples():
        
    # fetch dataset
    covertype = fetch_ucirepo(id=31)

    # data (as pandas dataframes)
    X = covertype.data.features
    y = covertype.data.targets

    # Extract 200000 samples
    X = X.sample(n=200000, random_state=1)
    y = y.loc[X.index]

    # metadata
    print(covertype.metadata)

    # variable information
    print(covertype.variables)

    # standardize X
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Check for missing values in X_scaled
    if X_scaled.isnull().values.any():
        print("Warning: X_scaled contains missing values.")
        X_scaled = X_scaled.dropna()

    # Check that all rows have the same number of columns
    num_cols = len(X_scaled.columns)
    for i, row in X_scaled.iterrows():
        if len(row) != num_cols:
            print(f"Warning: Row {i} has {len(row)} columns instead of {num_cols}")

    # write standardized X to a CSV file without headers
    X_scaled.to_csv('cover_type_x_200000.csv', index=False, header=False)

    # transform y
    mid_value = (y.iloc[:, 0].max() + y.iloc[:, 0].min()) / 2
    y_transformed = y.iloc[:, 0].map(lambda x: -1 if x < mid_value else 1)

    # Check for missing values in y_transformed
    if y_transformed.isnull().values.any():
        print("Warning: y_transformed contains missing values.")
        y_transformed = y_transformed.dropna()

    # write transformed y to a CSV file without headers
    y_transformed.to_csv('cover_type_y_200000.csv', index=False, header=False)



def generate_cover_1600_samples():
        
    # fetch dataset
    covertype = fetch_ucirepo(id=31)

    # data (as pandas dataframes)
    X = covertype.data.features
    y = covertype.data.targets

    # Extract 1600 samples
    X = X.sample(n=1600, random_state=1)
    y = y.loc[X.index]

    # metadata
    print(covertype.metadata)

    # variable information
    print(covertype.variables)

    # standardize X
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Check for missing values in X_scaled
    if X_scaled.isnull().values.any():
        print("Warning: X_scaled contains missing values.")
        X_scaled = X_scaled.dropna()

    # Check that all rows have the same number of columns
    num_cols = len(X_scaled.columns)
    for i, row in X_scaled.iterrows():
        if len(row) != num_cols:
            print(f"Warning: Row {i} has {len(row)} columns instead of {num_cols}")

    # write standardized X to a CSV file without headers
    X_scaled.to_csv('cover_type_x_1600.csv', index=False, header=False)

    # transform y
    mid_value = (y.iloc[:, 0].max() + y.iloc[:, 0].min()) / 2
    y_transformed = y.iloc[:, 0].map(lambda x: -1 if x < mid_value else 1)

    # Check for missing values in y_transformed
    if y_transformed.isnull().values.any():
        print("Warning: y_transformed contains missing values.")
        y_transformed = y_transformed.dropna()

    # write transformed y to a CSV file without headers
    y_transformed.to_csv('cover_type_y_1600.csv', index=False, header=False)

def generate_cover_3200_samples():
        
    # fetch dataset
    covertype = fetch_ucirepo(id=31)

    # data (as pandas dataframes)
    X = covertype.data.features
    y = covertype.data.targets

    # Extract 3200 samples
    X = X.sample(n=3200, random_state=1)
    y = y.loc[X.index]

    # metadata
    print(covertype.metadata)

    # variable information
    print(covertype.variables)

    # standardize X
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Check for missing values in X_scaled
    if X_scaled.isnull().values.any():
        print("Warning: X_scaled contains missing values.")
        X_scaled = X_scaled.dropna()

    # Check that all rows have the same number of columns
    num_cols = len(X_scaled.columns)
    for i, row in X_scaled.iterrows():
        if len(row) != num_cols:
            print(f"Warning: Row {i} has {len(row)} columns instead of {num_cols}")

    # write standardized X to a CSV file without headers
    X_scaled.to_csv('cover_type_x_3200.csv', index=False, header=False)

    # transform y
    mid_value = (y.iloc[:, 0].max() + y.iloc[:, 0].min()) / 2
    y_transformed = y.iloc[:, 0].map(lambda x: -1 if x < mid_value else 1)

    # Check for missing values in y_transformed
    if y_transformed.isnull().values.any():
        print("Warning: y_transformed contains missing values.")
        y_transformed = y_transformed.dropna()

    # write transformed y to a CSV file without headers
    y_transformed.to_csv('cover_type_y_3200.csv', index=False, header=False)


def generate_cover_6400_samples():
        
    # fetch dataset
    covertype = fetch_ucirepo(id=31)

    # data (as pandas dataframes)
    X = covertype.data.features
    y = covertype.data.targets

    # Extract 6400 samples
    X = X.sample(n=6400, random_state=1)
    y = y.loc[X.index]

    # metadata
    print(covertype.metadata)

    # variable information
    print(covertype.variables)

    # standardize X
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Check for missing values in X_scaled
    if X_scaled.isnull().values.any():
        print("Warning: X_scaled contains missing values.")
        X_scaled = X_scaled.dropna()

    # Check that all rows have the same number of columns
    num_cols = len(X_scaled.columns)
    for i, row in X_scaled.iterrows():
        if len(row) != num_cols:
            print(f"Warning: Row {i} has {len(row)} columns instead of {num_cols}")

    # write standardized X to a CSV file without headers
    X_scaled.to_csv('cover_type_x_6400.csv', index=False, header=False)

    # transform y
    mid_value = (y.iloc[:, 0].max() + y.iloc[:, 0].min()) / 2
    y_transformed = y.iloc[:, 0].map(lambda x: -1 if x < mid_value else 1)

    # Check for missing values in y_transformed
    if y_transformed.isnull().values.any():
        print("Warning: y_transformed contains missing values.")
        y_transformed = y_transformed.dropna()

    # write transformed y to a CSV file without headers
    y_transformed.to_csv('cover_type_y_6400.csv', index=False, header=False)

def generate_cover_12800_samples():
        
    # fetch dataset
    covertype = fetch_ucirepo(id=31)

    # data (as pandas dataframes)
    X = covertype.data.features
    y = covertype.data.targets

    # Extract 12800 samples
    X = X.sample(n=12800, random_state=1)
    y = y.loc[X.index]

    # metadata
    print(covertype.metadata)

    # variable information
    print(covertype.variables)

    # standardize X
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Check for missing values in X_scaled
    if X_scaled.isnull().values.any():
        print("Warning: X_scaled contains missing values.")
        X_scaled = X_scaled.dropna()

    # Check that all rows have the same number of columns
    num_cols = len(X_scaled.columns)
    for i, row in X_scaled.iterrows():
        if len(row) != num_cols:
            print(f"Warning: Row {i} has {len(row)} columns instead of {num_cols}")

    # write standardized X to a CSV file without headers
    X_scaled.to_csv('cover_type_x_12800.csv', index=False, header=False)

    # transform y
    mid_value = (y.iloc[:, 0].max() + y.iloc[:, 0].min()) / 2
    y_transformed = y.iloc[:, 0].map(lambda x: -1 if x < mid_value else 1)

    # Check for missing values in y_transformed
    if y_transformed.isnull().values.any():
        print("Warning: y_transformed contains missing values.")
        y_transformed = y_transformed.dropna()

    # write transformed y to a CSV file without headers
    y_transformed.to_csv('cover_type_y_12800.csv', index=False, header=False)


def generate_cover_25600_samples():
        
    # fetch dataset
    covertype = fetch_ucirepo(id=31)

    # data (as pandas dataframes)
    X = covertype.data.features
    y = covertype.data.targets

    # Extract 25600 samples
    X = X.sample(n=25600, random_state=1)
    y = y.loc[X.index]

    # metadata
    print(covertype.metadata)

    # variable information
    print(covertype.variables)

    # standardize X
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Check for missing values in X_scaled
    if X_scaled.isnull().values.any():
        print("Warning: X_scaled contains missing values.")
        X_scaled = X_scaled.dropna()

    # Check that all rows have the same number of columns
    num_cols = len(X_scaled.columns)
    for i, row in X_scaled.iterrows():
        if len(row) != num_cols:
            print(f"Warning: Row {i} has {len(row)} columns instead of {num_cols}")

    # write standardized X to a CSV file without headers
    X_scaled.to_csv('cover_type_x_25600.csv', index=False, header=False)

    # transform y
    mid_value = (y.iloc[:, 0].max() + y.iloc[:, 0].min()) / 2
    y_transformed = y.iloc[:, 0].map(lambda x: -1 if x < mid_value else 1)

    # Check for missing values in y_transformed
    if y_transformed.isnull().values.any():
        print("Warning: y_transformed contains missing values.")
        y_transformed = y_transformed.dropna()

    # write transformed y to a CSV file without headers
    y_transformed.to_csv('cover_type_y_25600.csv', index=False, header=False)


def generate_cover_51200_samples():
        
    # fetch dataset
    covertype = fetch_ucirepo(id=31)

    # data (as pandas dataframes)
    X = covertype.data.features
    y = covertype.data.targets

    # Extract 51200 samples
    X = X.sample(n=51200, random_state=1)
    y = y.loc[X.index]

    # metadata
    print(covertype.metadata)

    # variable information
    print(covertype.variables)

    # standardize X
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Check for missing values in X_scaled
    if X_scaled.isnull().values.any():
        print("Warning: X_scaled contains missing values.")
        X_scaled = X_scaled.dropna()

    # Check that all rows have the same number of columns
    num_cols = len(X_scaled.columns)
    for i, row in X_scaled.iterrows():
        if len(row) != num_cols:
            print(f"Warning: Row {i} has {len(row)} columns instead of {num_cols}")

    # write standardized X to a CSV file without headers
    X_scaled.to_csv('cover_type_x_51200.csv', index=False, header=False)

    # transform y
    mid_value = (y.iloc[:, 0].max() + y.iloc[:, 0].min()) / 2
    y_transformed = y.iloc[:, 0].map(lambda x: -1 if x < mid_value else 1)

    # Check for missing values in y_transformed
    if y_transformed.isnull().values.any():
        print("Warning: y_transformed contains missing values.")
        y_transformed = y_transformed.dropna()

    # write transformed y to a CSV file without headers
    y_transformed.to_csv('cover_type_y_51200.csv', index=False, header=False)


if __name__ == "__main__":
    # generate_cover_50000_samples()
    # generate_cover_100000_samples()
    # generate_cover_200000_samples()
    generate_cover_10000_samples()
    # generate_cover_1600_samples()
    # generate_cover_3200_samples()
    # generate_cover_6400_samples()
    # generate_cover_12800_samples()
    # generate_cover_25600_samples()
    # generate_cover_51200_samples()
    # generate_cover_samples()