import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
# Load the cencus.csv data
project_path = os.getcwd()
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
data = pd.read_csv(data_path)

# Split the data into train and test sets
train, test = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["salary"]
)
print(f"Train shape: {train.shape[0]}")
print(f"Test shape: {test.shape[0]}")
# Define the categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    train, # use the train dataset
    categorical_features=cat_features,
    label="salary",
    training=True # use training=True
    )

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train) # your code here

# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)

# load the model
model = load_model(
    model_path
) 

# Use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test) # your code here

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features
for col in cat_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            test,  # use the test dataset
            col,  # use the categorical feature
            slicevalue,  # use the unique value in the categorical feature
            cat_features,  # use the categorical features
            "salary",  # use the label
            encoder,  
            lb,  
            model
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
