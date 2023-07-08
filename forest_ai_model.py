from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import logging
import pandas as pd
from joblib import dump, load
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


class BaseAIModel:
    """
    A parent class for AI handling, with save, load, and encryption functions.

    Attributes:
        log_filename (str): filename for the log file.
        hex_encryption_key (str): the encryption key if there is one.
        logging_str_level (str): the logging level
    """

    # Constants used for encryption levels
    ENCRYPTION_SPLIT = 16
    ENCRYPTION_LENGTH = 32

    def __init__(self, log_filename: str, hex_encryption_key: str = None, logging_str_level: str = 'INFO') -> None:
        self.model = None
        self.trained_check = False

        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(log_filename)
        logging_level = logging.getLevelName(logging_str_level)
        handler.setLevel(logging_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.set_encryption_key(hex_encryption_key)
    
    def get_encryption_key(self) -> str:
        """
        Get the current encryption key.
        
        Returns:
            str: the encryption key in hex value
        """
        return self.encrypt_key.hex()
    
    def is_hex(self, s: str) -> bool:
        """
        Check if the given string is hexadecimal.
        
        Args:
            s (str): the string to be checked
            
        Returns:
            bool: boolean value for if the string is a hexadecimal
        """
        try:
            bytes.fromhex(s)
            return True
        except ValueError:
            return False
    
    def check_encryption_key(self, s: str)  -> bool:
        """
        Check if the provided encryption key is valid.
        
        Args:
            s (str): the string to be checked

        Returns:
            bool: boolean value for if the string is a hexadecimal and the right length
        """
        return len(s) in [32, 48, 64] and self.is_hex(s)
    
    def set_encryption_key(self, hex_encryption_key: str = None) -> None:
        """
        Set a new encryption key.
        
        Args:
            hex_encryption_key (str): the encryption key to implement
        """
        try:
            # If the encryption key is not None, attempt to implement it
            if hex_encryption_key is not None:
                if not self.check_encryption_key(hex_encryption_key):
                    self.logger.critical(f"Invalid key. Key length must be [32, 48, 64], key provided is {len(hex_encryption_key)}")
                    raise ValueError(f"Invalid key. Key length must be [32, 48, 64], key provided is {len(hex_encryption_key)}")
                self.encrypt_key = bytes.fromhex(hex_encryption_key)
                self.encrypt_check = True
            # If encryption key is set to None, disable encryption functions
            else:
                self.encrypt_key = None
                self.encrypt_check = False
        # Error catching
        except ValueError as e:
            self.logger.error(f"Invalid hexadecimal string provided for encryption key: {e}")
            self.encrypt_check = False
        except Exception as e:
            self.logger.error(f"Unexpected error occurred while setting the encryption key: {e}")
            self.encrypt_check = False
            raise e
        self.logger.info(f"Encryption set to {self.encrypt_check}")
    
    def new_encryption_key(self) -> str:
        """
        Generate a new encryption key.
        
        Returns:
            str: the newly generated encryption key
        """
        self.logger.debug("Creating new encryption key")
        self.encrypt_key = get_random_bytes(self.ENCRYPTION_LENGTH)
        return self.get_encryption_key()
    
    def save_model(self, filename: str) -> None:
        """
        Save the current model to a file.
        
        Args:
            filename (str): the name to save the file as
        """

        # If model hasnt been trained, cancel save
        if not self.get_trained_model_status():
            error_msg = "Failed to save model, model is not Trained"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
        # If encryption functions are enabled, switch to encrypted save
        if self.encrypt_check:
            self.save_encrypted_model(filename)
            return
        
        # Save the file
        try:
            dump(self.model, filename)
            self.logger.info(f"Model saved to {filename}")
        except FileNotFoundError as e:
            self.logger.error(f"File '{filename}' not found: {e}")
        except OSError as e:
            self.logger.error(f"OS error occurred while saving the model to '{filename}': {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error occurred while saving the model to '{filename}': {e}")
            raise e

    def save_encrypted_model(self, filename: str) -> None:
        """
        Save the current model to a file with encryption.
        
        Args:
            filename (str): the name to save the file as
        """

        # If model hasnt been trained, cancel save
        if not self.get_trained_model_status():
            error_msg = "Failed to save model, model is not Trained"
            self.logger.error(error_msg)
            raise Exception(error_msg)

        # If encryption funcstions are disabled, raise error
        if not self.encrypt_check or not self.check_encryption_key(self.encrypt_key):
            error_msg = f"save_encrypted_model called without correct parameters. Set the encryption key again to confirm details are correct."
            self.logger.error(error_msg)
            raise Exception(error_msg)

        # Encrypt bytes and save file
        try:
            model_bytes = io.BytesIO()
            dump(self.model, model_bytes)
            model_bytes = model_bytes.getvalue()

            cipher = AES.new(self.encrypt_key, AES.MODE_CBC)
            ciphered_data = cipher.iv + cipher.encrypt(pad(model_bytes, AES.block_size))

            with open(filename, 'wb') as f:
                f.write(ciphered_data)

            self.logger.info(f"Encrypted model saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save encrypted model to '{filename}': {e}")
            raise e

    def load_model(self, filename: str) -> None:
        """
        Load a model from a file.
        
        Args:
            filename (str): the path of the file to load in
        """

        # If model is already trainied, cancel load
        if self.get_trained_model_status():
            self.logger.warning("Unable to load model, model currently trained. Create a new instance to load in a model")
            return

        # If encryption functions are enabled, switch to encrypted load
        if self.encrypt_check:
            self.load_encrypted_model(filename)
            return
        
        # Load file and mark object as 'Trained"
        try:
            self.model = load(filename)
            self.logger.info(f"Model loaded from {filename}")
            self.trained_check = True
        except FileNotFoundError as e:
            self.logger.error(f"File '{filename}' not found: {e}")
        except OSError as e:
            self.logger.error(f"OS error occurred while loading the model from '{filename}': {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error occurred while loading the model from '{filename}': {e}")
            raise e
        

    def load_encrypted_model(self, filename: str) -> None:
        """
        Load a model from an encrypted file.
        
        Args:
            filename (str): the path of the file to load in
        """

        # If model is already trainied, cancel load
        if self.get_trained_model_status():
            self.logger.warning("Unable to load model, model currently trained. Create a new instance to load in a model")
            return
        
        # If encryption funcstions are disabled, raise error
        if not self.encrypt_check or not self.check_encryption_key(self.encrypt_key):
            error_msg = f"load_encrypted_model called without correct parameters. Set the encryption key again to confirm details are correct."
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
        # Load file, decrypt the bytes, and mark object as 'Trained'
        try:
            with open(filename, 'rb') as f:
                ciphered_data = f.read()
            
            iv = ciphered_data[:self.ENCRYPTION_SPLIT]
            ciphered_data = ciphered_data[self.ENCRYPTION_SPLIT:]
            cipher = AES.new(self.encrypt_key, AES.MODE_CBC, iv=iv)
            model_bytes = unpad(cipher.decrypt(ciphered_data), AES.block_size)

            self.model = load(io.BytesIO(model_bytes))
            self.logger.info(f"Encrypted model loaded from {filename}")
            self.trained_check = True
        except Exception as e:
            self.logger.error(f"Failed to load encrypted model '{filename}': {e}")
            raise e

    def get_default_param_grid(self) -> dict:
        """
        Retrieve the default parameter grid for grid searching the best parameters.
        
        Returns:
            dict: default grid for doing a grid search on RandomForest"""
        self.logger.debug("retreiving default param grid")
        default_param_grid = {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt'],
            'bootstrap': [True, False]
        }
        return default_param_grid
    
    def get_trained_model_status(self) -> bool:
        """
        Check the trained status of the model.
        
        Returns:
            bool: boolean on whether the model has been trained or not
        """
        return self.model is not None and self.trained_check
    
    def split_training_data(self, X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2, random_state: int = 1) -> tuple:
        """
        Split the provided data into training and testing sets.
        
        Args:
            X (pd.DataFrame): the X component of the training data
            y (pd.DataFrame): the y component of the training data
            test_size (float): the ratio of testing data out of the whole (1)
            random_state (int): sets the random number generator seed
        
        Returns:
            X_train (pd.DataFrame): the X component set for training the model
            X_test (pd.DataFrame): the X component set for testing the model
            y_train (pd.DataFrame): the y component set for training the model
            y_test (pd.DataFrame): the y component set for testing the model
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    
    def normalize_data(self, X: pd.DataFrame)  -> pd.DataFrame:
        """
        Normalize the provided data.
        
        Args:
            X (pd.DataFrame): the X component to be normalized
            
        Returns:
            pd.DataFrame: the X component normalized"""
        self.logger.debug("Normalizing X frame")
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)


class RandomForestModelSingleOutput(BaseAIModel):
    """
    Uses the RandomForestClassification in the single context for creating models.

    A child of the BaseAIModel class.
    """
    def __init__(self, log_filename: str, hex_encryption_key: str = None, logging_str_level: str = 'INFO') -> None:
        super().__init__(log_filename=log_filename, hex_encryption_key=hex_encryption_key, logging_str_level=logging_str_level)
        self.logger.info("RandomForestModelSingleOutput instance created")
        
    def quick_train(self, X: pd.DataFrame, y: pd.DataFrame, chosen_params: dict = None, normalize: bool = True) -> None:
        """
        Quickly train the model with the provided data and optional parameters.
        
        Args:
            X (pd.DataFrame): the X component of the model training
            y (pd.DataFrame): the y component of the model training
            chosen_params (dict): optional parameters to set the model training to
            normalize (bool): whether or not the X component should be normalized
        """

        # If model is already trained, return
        if self.get_trained_model_status():
            self.logger.warning("Unable to train model, model currently trained. Create a new instance to train a model")
            return
        
        try:
            self.logger.info("Training single output model")

            # Create model with either provided parameters, or default parameters
            if chosen_params is not None:
                self.model = RandomForestClassifier(**chosen_params, random_state=1)
            else:
                self.model = RandomForestClassifier(random_state=1)

            # If normalization is required, applies normalization to the X
            if normalize:
                X = self.normalize_data(X)
            X_train, X_test, y_train, y_test = self.split_training_data(X, y)

            # Trains model on training dataset, and creates prediction based on X test dataset data
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            # Scores accuracy on the prediction verses actual y test dataset data, and sets trained check to True
            accuracy = accuracy_score(y_test, y_pred)
            self.logger.info(f"Training complete. Accuracy: {accuracy}")
            self.trained_check = True
        except ValueError as e:
            self.logger.error(f"Failed to train model: {e}")
    
    def best_fit_train(self, X: pd.DataFrame, y: pd.DataFrame, param_grid: dict = None, normalize: bool = True) -> None:
        """
        Train the model with the provided data against a grid of parameters.
        
        Args:
            X (pd.DataFrame): the X component of the model training
            y (pd.DataFrame): the y component of the model training
            param_grid (dict): the grid of parameters to test the model against. If this is left as None, then the default model is chosen
            normalize (bool): whether or not the X component should be normalized
        """

        # If model is already trained, return
        if self.get_trained_model_status():
            self.logger.warning("Unable to train model, model currently trained. Create a new instance to train a model")
            return
        
        try:
            self.logger.info("Training single output model")

            # Create model with default parameters
            self.model = RandomForestClassifier(random_state=1)

            # If normalization is required, applies normalization to the X
            if normalize:
                X = self.normalize_data(X)
            X_train, X_test, y_train, y_test = self.split_training_data(X, y)

            # If no parameter grid is provided, use the default one
            if param_grid is None:
                param_grid = self.get_default_param_grid()

            # Perform a gridsearch of the param grid to find the best model
            grid_search = GridSearchCV(self.model, param_grid, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            # Get best model based on accuracy, and set trained check to True
            try:
                self.model = grid_search.best_estimator_
                y_pred = self.model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                self.logger.info(f"Training complete. Best parameters: {grid_search.best_params_}. Accuracy: {accuracy}")
                self.trained_check = True
            except Exception as e:
                self.logger.warning(f"Unable to set ideal model: {e}")
        except ValueError as e:
            self.logger.error(f"Failed to train model: {e}")

    def predict_response(self, df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
        """
        Predict the response for the provided data.
        
        Args:
            df (pd.DataFrame): the X component for the model to respond to.
            normalize (bool): whether or not the X component should be normalized.

        Returns:
            pd.DataFrame: the y component generated by the model
        """

        # If the model doesnt exist, raise an error
        if not self.get_trained_model_status():
            error_msg = "Unable to predict response as model is currently empty. Load an existing model or train a new one"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
        # Predict the result based on the provided data, normalizing if necessary
        try:
            if normalize:
                df = self.normalize_data(df)
            y_pred = self.model.predict(df)
            y_pred_df = pd.DataFrame(y_pred, columns=['prediction'])
            self.logger.info("Predictions made")
            return y_pred_df
        except Exception as e:
            self.logger.error(f"Failed to predict responses. {e}")


class RandomForestModelMultiOutput(BaseAIModel):
    """
    Uses the RandomForestClassification in the multi context for creating models.

    A child of the BaseAIModel class.
    """
    def __init__(self, log_filename: str, hex_encryption_key: str = None, logging_str_level: str = 'INFO') -> None:
        super().__init__(log_filename=log_filename, hex_encryption_key=hex_encryption_key, logging_str_level=logging_str_level)
        self.logger.info("RandomForestModelMultiOutput instance created")

    def quick_train(self, X: pd.DataFrame, Y: pd.DataFrame, chosen_params: dict = None, normalize: bool = True) -> None:
        """
        Quickly train the model with the provided data and optional parameters.
        
        Args:
            X (pd.DataFrame): the X component of the model training
            Y (pd.DataFrame): the Y component of the model training
            chosen_params (dict): optional parameters to set the model training to
            normalize (bool): whether or not the X component should be normalized
        """

        # If model is already trained, return
        if self.get_trained_model_status():
            self.logger.warning("Unable to train model, model currently trained. Create a new instance to train a model")
            return
        
        try:
            self.logger.info("Training multi-output model")

            # Create model with either provided parameters, or default parameters
            if chosen_params is not None:
                base_model = RandomForestClassifier(**chosen_params, random_state=1)
                self.model = MultiOutputClassifier(base_model)
            else:
                self.model = MultiOutputClassifier(RandomForestClassifier(random_state=1))

            # If normalization is required, applies normalization to the X
            if normalize:
                X = self.normalize_data(X)
            X_train, X_test, Y_train, Y_test = self.split_training_data(X, Y)

            # Trains model on training dataset, and creates prediction based on X test dataset data
            self.model.fit(X_train, Y_train)
            Y_pred = self.model.predict(X_test)

            # Scores accuracy on the prediction verses actual Y test dataset data, and sets trained check to True
            f1_score_micro = f1_score(Y_test, Y_pred, average='micro')
            self.logger.info(f"Training complete. F1 Score (micro): {f1_score_micro}")
            self.trained_check = True
        except ValueError as e:
            self.logger.error(f"Failed to train model: {e}")

    def best_fit_train(self, X: pd.DataFrame, Y: pd.DataFrame, param_grid: dict = None, normalize: bool = True) -> None:
        """
        Train the model with the provided data against a grid of parameters.
        
        Args:
            X (pd.DataFrame): the X component of the model training
            Y (pd.DataFrame): the Y component of the model training
            param_grid (dict): the grid of parameters to test the model against. If this is left as None, then the default model is chosen
            normalize (bool): whether or not the X component should be normalized
        """

        # If model is already trained, return
        if self.get_trained_model_status():
            self.logger.warning("Unable to train model, model currently trained. Create a new instance to train a model")
            return
        
        try:
            self.logger.info("Training multi-output model")

            # Create model with default parameters
            self.model = MultiOutputClassifier(RandomForestClassifier(random_state=1))

            # If normalization is required, applies normalization to the X
            if normalize:
                X = self.normalize_data(X)
            X_train, X_test, Y_train, Y_test = self.split_training_data(X, Y)

            # If no parameter grid is provided, use the default one
            if param_grid is None:
                param_grid = self.get_default_param_grid()

            # Perform a gridsearch of the param grid to find the best model
            grid_search = GridSearchCV(self.model, param_grid, cv=5, verbose=1, n_jobs=-1, scoring='f1_micro')
            grid_search.fit(X_train, Y_train)

            # Get best model based on accuracy (f1 micro), and set trained check to True
            try:
                self.model = grid_search.best_estimator_
                Y_pred = self.model.predict(X_test)
                f1_score_micro = f1_score(Y_test, Y_pred, average='micro')
                self.logger.info(f"Training complete. Best parameters: {grid_search.best_params_}. F1 score (micro): {f1_score_micro}")
                self.trained_check = True
            except Exception as e:
                self.logger.warning(f"Unable to set ideal model: {e}")           
        except ValueError as e:
            self.logger.error(f"Failed to train model: {e}")

    def predict_response(self, df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
        """
        Predict the response for the provided data.
        
        Args:
            df (pd.DataFrame): the X component for the model to respond to.
            normalize (bool): whether or not the X component should be normalized.

        Returns:
            pd.DataFrame: the Y component generated by the model
        """

        # If the model doesnt exist, raise an error
        if not self.get_trained_model_status():
            error_msg = "Unable to predict response as model is currently empty. Load an existing model or train a new one"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
        # Predict the result based on the provided data, normalizing if necessary and creating columns = different reponses count
        try:
            if normalize:
                df = self.normalize_data(df)
            y_pred = self.model.predict(df)
            column_names = [f"prediction{i+1}" for i in range(y_pred.shape[1])]
            y_pred_df = pd.DataFrame(y_pred, columns=column_names)
            self.logger.info("Predictions made")
            return y_pred_df
        except Exception as e:
            self.logger.error(f"Failed to predict responses. {e}")