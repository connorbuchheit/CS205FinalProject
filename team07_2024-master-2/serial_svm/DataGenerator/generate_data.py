import os
import numpy as np
import pandas as pd

def generate_data(n, m):
    # Generate n samples with m dimensions from a Gaussian distribution
    X = np.round(np.random.normal(0, 1, (n, m)), 4)
    
    # Generate n labels (either 0 or 1)
    y = np.random.randint(2, size=n)
    
    return X, y

def save_data(X, y, n, m):
    # Create the directory if it does not exist
    if not os.path.exists('Data'):
        os.makedirs('Data')

    # Save the generated data and labels into separate CSV filesp
    pd.DataFrame(X).to_csv(f"Data/data_{n}_{m}_X.csv", index=False)
    pd.DataFrame(y).to_csv(f"Data/data_{n}_{m}_y.csv", index=False)

def generate_and_save_data(n, m):
    # Generate data and labels
    X, y = generate_data(n, m)
    
    # Save data and labels
    save_data(X, y, n, m)

if __name__ == "__main__":

    # Generate and save data for different (n, m) values to test scaling
    generate_and_save_data(100, 10)
    generate_and_save_data(1000, 20)
    generate_and_save_data(10000, 30)
    generate_and_save_data(1000000, 5000)
