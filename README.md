# Neural Network Approach on Modeling Customer Lifetime Value

This repository is the online appendix to the master's thesis, available [here](#) [TODO: add link once available].

The modeling pipeline is organized as an API with six steps:
```python
import lstm_clv

EXPERIMENT = "example"

# 1. initiate folder structure for a model application and create config with pipeline parameters (stored on disc)
lstm_clv.make_config(experiment_name=EXPERIMENT, **config_parms)

# 2. prepare data into format expected by the model
lstm_clv.prepare_data(EXPERIMENT)

# 3. make predictions using the extended Pareto/NBD model
lstm_clv.evaluate_benchmark(EXPERIMENT)

# 4. optimize hyperparameters for the LSTM-CLV model (will update config with optimal values)
grid_search = lstm_clv.ParameterGridSearch(EXPERIMENT, **parms)
grid_search.optimize(...)

# 5. train the LSTM-CLV model, predict CLV, evaluate
lstm_clv.train_lstm(EXPERIMENT)

# 6. predict CLV, evaluate performance
lstm_clv.evaluate_lstm(EXPERIMENT)
```

Notebooks `cdnow.ipynb` and `fintech.ipynb` contain modeling results as described in the thesis. Raw data for the CDNow application can be found in the `data-cdnow` folder.

It is possible to apply the LSTM-CLV model on different data. Input data should have the following structure:
* `transactions.csv`: `user_id`, `ts` (e.g. weeks), `event` (referred to as product in the thesis), `value`
* `user_attributes.csv`: `user_id`, ... (attribute columns)
* `margins.json`: `{event: margin}`

Dependencies required to reproduce the modeling pipeline are outlaid in `pyproject.toml` and can be resolved using the [poetry](https://python-poetry.org/) package. 
