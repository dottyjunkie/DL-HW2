import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def train_test_split(raw, random_state=42):
    """
    Parameters:

    Returns:
    """
    entries = raw.columns

    want = raw[entries]

    train = want.sample(frac=.8, random_state=random_state)
    y_train = train[['Activities_Types']]
    X_train = train.drop(columns=['Activities_Types'])
    # X_train = (X_train - X_train.mean()) / X_train.std()

    test = want.drop(train.index)
    y_test = test[['Activities_Types']]
    X_test = test.drop(columns=['Activities_Types'])
    # X_test = (X_test - X_test.mean()) / X_test.std()

    return (X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    raw = pd.read_csv('Data.csv')
    X_train, X_test, y_train, y_test = train_test_split(raw)