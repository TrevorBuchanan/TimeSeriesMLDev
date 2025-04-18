from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from ml.lstm.lstm import LSTMModel


class StackedLSTM(LSTMModel):
    def __init__(self, n_features: int, n_steps_in: int, n_steps_out: int,
                 n_epochs: int = 500, verbose: int = 1, saved_model_num=None) -> None:
        super().__init__(n_features, n_steps_in, n_steps_out, n_epochs, verbose, saved_model_num)

    def _build_model(self) -> None:
        l2_reg = l2(0.001)

        self.model = Sequential()
        self.model.add(Input(shape=(self.n_steps_in, self.n_features)))
        # self.model.add(Masking(mask_value=0.0))  # if padding is used

        self.model.add(LSTM(256, return_sequences=True, kernel_regularizer=l2_reg))
        self.model.add(Dropout(0.3))
        self.model.add(BatchNormalization())

        self.model.add(LSTM(256, return_sequences=True, kernel_regularizer=l2_reg))
        self.model.add(Dropout(0.3))
        self.model.add(BatchNormalization())

        self.model.add(LSTM(128, kernel_regularizer=l2_reg))
        self.model.add(Dense(self.n_steps_out, activation='linear', kernel_regularizer=l2_reg))

        optimizer = Adam(learning_rate=0.0005)
        self.model.compile(optimizer=optimizer, loss='mse')
