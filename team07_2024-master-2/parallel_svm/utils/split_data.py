#split_data.py 

"""
This tool takes in some csv data and splits it into a training set and a testing set

How to use:

Assuming you are at base directory

$ python utils/split_data.py [X_data.csv] [Y_data.csv] [Percentage-of-data-to-be-train_set]

example: python3 split_data.py ../Data/cover_type_x_1000.csv ../Data/cover_type_y_1000.csv 0.75

The training set will be save in [X_data_tr.csv] [Y_data_tr.csv]  and Test data in in [X_data_te.csv] [Y_data_te.csv]
"""

import os,sys
import pandas as pd
from sklearn.model_selection import train_test_split

def main(argv = sys.argv):
    # Check if the number of arguments is correct
    if len(argv) != 4:
        print("Usage: python split_data.py [X_data.csv] [Y_data.csv] [Percentage-of-data-to-be-train_set]")
        return

    # Get the file names and the split ratio from the command line arguments
    X_data_file = argv[1]
    Y_data_file = argv[2]
    split_ratio = float(argv[3])

    # Load the data from the csv files
    X_data = pd.read_csv(X_data_file)
    Y_data = pd.read_csv(Y_data_file)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, train_size=split_ratio, random_state=42)

    # Save the training and testing sets to csv files
    X_train.to_csv(X_data_file.replace('.csv', '_tr.csv'), index=False ,header=False)
    Y_train.to_csv(Y_data_file.replace('.csv', '_tr.csv'), index=False ,header=False)
    X_test.to_csv(X_data_file.replace('.csv', '_te.csv'), index=False ,header=False)
    Y_test.to_csv(Y_data_file.replace('.csv', '_te.csv'), index=False ,header=False)

    print("The data has been successfully split into training and testing sets.")

if __name__ == '__main__':
    main(sys.argv)
