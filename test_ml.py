import pytest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import compute_model_metrics, train_model
from train_model import train_test_split

# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_type_metrics():
    """
    This function tests the data type of the returned values of the compute_model_metrics function.
    It checks if the precision, recall, and fbeta values weather are floats.
    The test uses a sample input of y and preds arrays.
    """
    # Sample input y and preds arrays
    y = np.array([0, 1, 0, 1, 0, 1])
    preds = np.array([0, 1, 1, 1, 0, 0])

    # Call the function to compute metrics
    precision, recall, fbeta = compute_model_metrics(y, preds)

    # Check the type of the returned values
    assert isinstance(precision, float), "The precision is not a float"
    assert isinstance(recall, float), "The recall is not a float"
    assert isinstance(fbeta, float), "The fbeta is not a float"



# TODO: implement the second test. Change the function name and input as needed
def test_train_model():
    """
    The function tests the behind algorithm of the trained model.
    It checks if the model is of type RandomForestClassifier.
    The test uses a sample input of X_train and y_train arrays.
    """

    # Sample input X_train and y_train arrays
    X_train = np.random.rand(10, 5)
    y_train = np.random.randint(0, 2, size=10)

    # Train the model based on the sample input
    model = train_model(X_train, y_train)

    # Check the type of the model
    assert isinstance(model, RandomForestClassifier),"The model is not of type RandomForestClassifier"
    


# TODO: implement the third test. Change the function name and input as needed
def test_datasets():
    """
    The test to ensure the ammount of data in the training and testing datasets. which is supposed
    to be 80% and 20% respectively.
    The test uses the sample dataframe input of 100 rows and 3 columns.
    To test if training dataset is 80 rows and testing dataset is 20 rows.
    """
    # Sample input dataframe with 100 rows and 3 columns
    data = pd.DataFrame({
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
        "salary": ["<=50K"] * 70 + [">50K"] * 30
    })
    # Split the data into train and test sets
    train, test = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["salary"]
    )
    # Check the shape of the train and test datasets
    assert len(train) == 80, f"Expected 80 rows in training dataset, got {len(train)}"
    assert len(test) == 20, f"Expected 20 rows in testing dataset, got {len(test)}"

    
    
