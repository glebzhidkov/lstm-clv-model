from typing import Callable, Dict

from tensorflow.keras import Model, layers


def build_lstm_model(
    nr_events: int,
    nr_user_attributes: int,
    window: int,
    lstm_units: int,
) -> Model:
    """
    input_lstm -> lstm -> dense -> output
    """
    input_lstm = layers.Input(shape=(window, nr_events))
    input_history = layers.Input(shape=(nr_events,))  # not used
    input_user = layers.Input(shape=(nr_user_attributes,))  # not used

    lstm = layers.LSTM(units=lstm_units)(input_lstm)
    dense_lstm = layers.Dense(units=nr_events)(lstm)

    model = Model(inputs=[input_lstm, input_history, input_user], outputs=dense_lstm)
    return model


def build_lstm_model_with_history(
    nr_events: int,
    nr_user_attributes: int,
    window: int,
    lstm_units: int,
) -> Model:
    """
    input_lstm -> lstm -> dense -> concatenate -> dense -> output
    input_history --------------/
    """
    input_lstm = layers.Input(shape=(window, nr_events))
    input_history = layers.Input(shape=(nr_events,))
    input_user = layers.Input(shape=(nr_user_attributes,))  # not used

    lstm = layers.LSTM(units=lstm_units)(input_lstm)
    dense_lstm = layers.Dense(units=nr_events)(lstm)

    concat = layers.Concatenate()([dense_lstm, input_history])
    dense_out = layers.Dense(units=nr_events)(concat)

    model = Model(inputs=[input_lstm, input_history, input_user], outputs=dense_out)
    return model


def build_lstm_model_with_history_and_user_info(
    nr_events: int,
    nr_user_attributes: int,
    window: int,
    lstm_units: int,
) -> Model:
    """
    input_lstm -> lstm -> dense -> concatenate -> dense -> output
    input_history --------------/
    input_user -> dense -------/
    """
    input_lstm = layers.Input(shape=(window, nr_events))
    input_history = layers.Input(shape=(nr_events,))
    input_user = layers.Input(shape=(nr_user_attributes,))

    lstm = layers.LSTM(units=lstm_units)(input_lstm)
    dense_lstm = layers.Dense(units=nr_events, activation="tanh")(lstm)

    dense_user = layers.Dense(units=nr_events, activation="tanh")(input_user)

    concat = layers.Concatenate()([dense_lstm, input_history, dense_user])
    dense_out = layers.Dense(units=nr_events, activation="tanh")(concat)

    model = Model(inputs=[input_lstm, input_history, input_user], outputs=dense_out)
    return model


MODEL_BUILDERS: Dict[int, Callable[..., Model]] = {
    1: build_lstm_model,
    2: build_lstm_model_with_history,
    3: build_lstm_model_with_history_and_user_info,
}


def build_model(
    model_type: int,
    nr_events: int,
    nr_user_attributes: int,
    window: int,
    lstm_units: int,
) -> Model:
    """Construct the LSTM model"""
    builder = MODEL_BUILDERS[model_type]
    return builder(
        nr_events=nr_events,
        nr_user_attributes=nr_user_attributes,
        window=window,
        lstm_units=lstm_units,
    )
