import pickle # Use to save and load the model
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
from sklearn.ensemble import RandomForestClassifier # Use to train the random forest model


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # initialize the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Training model...") 

    # fit the model
    model.fit(X_train, y_train)
    print("Model trained.")

    # return the model
    return model
    


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    
    # run the model on the input data
    preds = model.predict(X)

    # return the predictions
    return preds

    

def save_model(model, path):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    # serialize the model to a pickle file
    with open(path, "wb") as f:
        # save the model to the file
        pickle.dump(model, f)
        # print the save status
        print(f"Model saved to {path}")


def load_model(path):
    """ Loads pickle file from `path` and returns it."""

    # load the model from the pickle file
    with open(path, "rb") as f:
        # return the saved data in the file
        return pickle.load(f)
    


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model):
    """ Computes the model metrics on a slice of the data specified by a column name and 
    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : ???
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """

    # TODO: implement the function
    X_slice, y_slice, _, _ = process_data(
        X = data[data[column_name] == slice_value], # Slice the data based on the provided column and value
        categorical_features=categorical_features,
        label=label,
        training=False, # use training = False
        encoder=encoder,
        lb=lb
    )
    preds = model.predict(X_slice) # get prediction on X_slice using the inference function
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
