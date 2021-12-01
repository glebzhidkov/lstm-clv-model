import os
from os.path import join as join_paths


class ExperimentPaths:
    """
    Utility class to standardize how files are stored for all steps of the pipeline
    """

    def __init__(self, experiment: str):
        self._experiment = experiment
        self.root = join_paths("experiments", experiment)
        self.config = join_paths(self.root, "config.json")
        self.data = join_paths(self.root, "data")
        self.results = join_paths(self.root, "results")
        self.scalers = join_paths(self.root, "scalers")

        self.user_attrs = join_paths(self.data, "user_attributes.csv")
        self.transactions_matrix = join_paths(self.data, "transactions.npz")
        self.data_for_lstm = join_paths(self.data, "data_for_lstm.npz")

        self.user_attrs_scaler = join_paths(self.scalers, "user_attrs.csv")
        self.transactions_scaler = join_paths(self.scalers, "transactions.csv")

        self.model_checkpoint = join_paths(self.root, "checkpoint.h5")
        self.margins = join_paths(self.data, "margins.json")

        self.lstm_preds = join_paths(self.results, "lstm_preds.csv")
        self.lstm_performance = join_paths(self.results, "lstm_perform.json")
        self.benchmark_preds = join_paths(self.results, "benchmark_preds.csv")
        self.benchmark_performance = join_paths(self.results, "benchmark_perform.json")
        self.parameter_grid = join_paths(self.results, "parameter_grid.txt")

    def initiate_folders(self):
        folders = ("experiments", self.root, self.data, self.results, self.scalers)
        for path in folders:
            if not os.path.exists(path):
                os.mkdir(path)
