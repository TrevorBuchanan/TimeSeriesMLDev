import numpy as np

from tensorflow.keras.models import load_model

from ml.lstm.lstm_utils import split_sequence, split_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class LSTMModel:
    def __init__(self, n_features: int, n_steps_in: int, n_steps_out: int,
                 n_epochs: int = 500, verbose: int = 1, saved_model_num=None) -> None:
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.model = None
        self.n_features = n_features  # Dimensionality of data at each time step.
        self.n_steps_in = n_steps_in  # Number of time steps used as input.
        self.n_steps_out = n_steps_out  # Number of time steps the model predicts.
        if not saved_model_num:
            self._build_model()
        else:
            self._load_trained_model(saved_model_num)

    def _build_model(self) -> None:
        """
        Builds an LSTM model and sets self.model = keras.models.Sequential - New built LSTM model
        :return: None
        """
        raise NotImplementedError("No build_model function implemented")

    def _load_trained_model(self, model_num):
        self.model = load_model(f'ml/lstm/saved_models/model_{model_num}.keras')

    def save_trained_model(self, model_num):
        self.model.save(f'ml/lstm/saved_models/model_{model_num}.keras')

    def train(self, input: list) -> None:
        """
        Used to train LSTM on given training sequences targeting corresponding given target values
        :param input: list - list of sequences that correspond to a target output in target_values
        (last seq is sequent to predict)
        :return: None
        """
        if not self.model:
            print("No model exists to train. ")
            return
        if len(input) > 1:
            # list of lists -> multivariate
            # convert raw_input to [rows, columns] structure
            for i, array_seq in enumerate(input):
                input[i] = array_seq.reshape((len(array_seq), 1))
            dataset = np.hstack(input)
            X, y = split_sequences(dataset, self.n_steps_in, self.n_steps_out)
        else:
            # list of nums -> univariate
            X, y = split_sequence(input[0], self.n_steps_in, self.n_steps_out)  # Use first (only) list
            X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        # self.n_features = X.shape[2]  # --> Not needed, because n_features is already given to class at initiation
        # callbacks = [
        #     EarlyStopping(patience=20, restore_best_weights=True),
        #     ReduceLROnPlateau(patience=10, factor=0.5),
        # ]
        # self.model.fit(X, y, epochs=self.n_epochs, callbacks=callbacks, verbose=self.verbose)
        self.model.fit(X, y, epochs=self.n_epochs, verbose=self.verbose)

    def predict(self, sequence):
        """
        Function to be called to get predicted value after training
        :param sequence: list - Sequence of numbers to be tested and predicted for
        :return: Result of running the input sequence through the LSTM model
        """

        # TODO: Add check for if sequence is in the correct form

        # Debug prints
        # print("Input")
        # print(sequence)

        sequence = np.array([sequence])
        x_input = sequence.reshape((1, self.n_steps_in, self.n_features))
        predicted = self.model.predict(x_input, verbose=self.verbose)
        result = [float(num) for num in predicted[0]]

        # Debug prints
        # print("Result")
        # print(result)

        return result
