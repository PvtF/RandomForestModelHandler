# Random Forest Models - Single and Multi Output

This project provides a framework for quickly building and training single-output and multi-output models using RandomForest classification. The code base is built on the principles of Object Oriented Programming, with well-documented classes and methods for easy use and modification. This framework allows for quick training, best-fit training through grid search, normalization of data, and prediction of responses for new data.

## BaseAImodel

The BaseAIModel class is the superclass for RandomForestModelSingleOutput and RandomForestModelMultiOutput classes. It provides foundational methods for handling model save/load operations and data preprocessing. It also integrates functionalities for logging and encryption, making it a great base for any machine learning model.

Key methods in the BaseAIModel class include:
1. **save_model:** This method saves the trained model to a file. If encryption is enabled and an encryption key is set, it saves the model in an encrypted form. The filename is passed as an argument.
2. **load_model:** This method loads a trained model from a file. If encryption is enabled and an encryption key is set, it can load the model from an encrypted file. The filename is passed as an argument.
3. **normalize_data:** This method accepts a pandas DataFrame (X) and returns a normalized DataFrame using MinMaxScaler.
4. **split_training_data:** This method splits the provided data into training and testing sets. The test size and random seed can be set as arguments.
5. **get_default_param_grid:** This method returns the default parameter grid for hyperparameter tuning in a grid search.
6. set_encryption_key, new_encryption_key, get_encryption_key: These methods are used to handle the encryption key for encrypted model save/load operations.

Note that the object will only encrypt/decrypt if an encryption key is set.

### Example usage

Import the **BaseAIModel** class, instantiate it, and call the appropriate methods.

```python
from path.to.module import BaseAIModel

# Initialize the BaseAIModel
base_model = BaseAIModel(log_filename='base_model.log')

# Suppose you have trained a model in the base_model instance
# You can save the model using
base_model.save_model('base_model.joblib')

# You can load the saved model using
base_model.load_model('base_model.joblib')

# Normalize the data
normalized_data = base_model.normalize_data(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = base_model.split_training_data(X, y)

# Set the encryption key for secure model save/load operations
base_model.set_encryption_key('your_hex_encryption_key')

# Generate a new encryption key
new_key = base_model.new_encryption_key()

# Get the default grid for grid-search model generation
default_grid = base_model.get_default_param_grid()
```

## RandomForestModelSingleOutput

The **RandomForestModelSingleOutput** class is designed for classification problems where the target has a single output (either binary or multiclass classification). It inherits from the **BaseAIModel** class, and extends its functionalities by implementing the following methods:

1. **quick_train**: This method accepts the input features (X) and target (y) along with optional hyperparameters and a boolean flag for normalization. If the model is already trained, this method logs a warning and returns. It trains the RandomForestClassifier using the provided data and hyperparameters, and then logs the accuracy of the model on a test set.
2. **best_fit_train**: This method performs a grid search over a range of hyperparameters to find the optimal parameters for the RandomForestClassifier. It takes the input features (X), target (y), a parameter grid, and a boolean flag for normalization. If the model is already trained, it logs a warning and returns.
3. **predict_response**: This method predicts the target variable based on the input features using the trained model. It raises an exception if the model is not yet trained.

## RandomForestModelMultiOutput

The **RandomForestModelMultiOutput** class is designed for classification problems where the target has multiple outputs. Similar to **RandomForestModelSingleOutput**, it inherits from the **BaseAIModel** class and implements **quick_train**, **best_fit_train**, and **predict_response** methods. The main difference is that it wraps the RandomForestClassifier in a MultiOutputClassifier, and uses the F1 score (micro-averaged) as the evaluation metric.

## Prerequisites

To use this project, you need to have Python 3.7 or later installed. This project uses the following Python packages, which you should install in your Python environment:

- pandas
- sklearn
- joblib
- crypto

You can install these from the requirements file

```bash
pip install -r requirements.txt
```

## How to Use

Import the necessary class (either **RandomForestModelSingleOutput** or **RandomForestModelMultiOutput**), instantiate it, and call the appropriate methods.

```python
from path.to.module import RandomForestModelSingleOutput, RandomForestModelMultiOutput

# Single Model
single_model = RandomForestModelSingleOutput(log_filename='training.log')
single_model.quick_train(X, y, chosen_params={'n_estimators': 100}, normalize=True)
predictions = single_model.predict_response(single_test_data, normalize=True)

# Multi Model
multi_model = RandomForestModelMultiOutput(log_filename='training.log')
multi_model.quick_train(X, Y, chosen_params={'n_estimators': 100}, normalize=True)
predictions = multi_model.predict_response(multi_test_data, normalize=True)
```

