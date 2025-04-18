from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from ml.lstm.lstm import LSTMModel
from ml.lstm.lstm_utils import split_sequence


class VanillaLSTM(LSTMModel):
    def __init__(self, n_features: int, n_steps_in: int, n_steps_out: int,
                 n_epochs: int = 500, verbose: int = 1, saved_model_num=None) -> None:
        super().__init__(n_features, n_steps_in, n_steps_out, n_epochs, verbose, saved_model_num)

    def _build_model(self) -> None:
        self.model = Sequential()
        self.model.add(Input(shape=(self.n_steps_in, self.n_features)))
        self.model.add(LSTM(100, activation='relu'))
        self.model.add(Dense(self.n_steps_out))
        optimizer = Adam(learning_rate=0.0005)
        self.model.compile(optimizer=optimizer, loss='mse')
