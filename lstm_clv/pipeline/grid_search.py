import itertools
import random
from typing import Any, Dict, List, Tuple

import pandas as pd
from lstm_clv.libs import Config, ExperimentPaths, get_logger
from lstm_clv.pipeline.lstm_model import LstmClvModel
from lstm_clv.pipeline.preparation import DataPreparationWorker
from pathos.helpers import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool

logger = get_logger("grid_search")
Parms = Dict[str, List[Any]]


class ParameterGridSearch:
    """
    Utility class to find optimal hyperparameters
    """

    def __init__(
        self,
        experiment: str,
        lstm_window: List[int],
        lstm_units: List[int],
        learning_rate: List[float],
        batch_size: List[float],
    ):
        self.paths = ExperimentPaths(experiment)

        self._experiment = experiment
        self._config = Config.load(self.paths.config)

        parms = dict(
            lstm_window=lstm_window,
            lstm_units=lstm_units,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        self._grid = self._permutate_dict_of_lists(parms)

    def optimize(
        self,
        max_combinations: int,
        target: str = "val_loss",
    ) -> pd.DataFrame:
        # minimising target

        config = self._config
        with open(self.paths.parameter_grid, "w") as log:
            log.write("")

        def train_lstm(parms: Dict[str, Any]):

            lstm_window = parms["lstm_window"]
            lstm_units = parms["lstm_units"]
            learning_rate = parms["learning_rate"]
            batch_size = parms["batch_size"]

            data_preparation_worker = DataPreparationWorker(self.paths)
            data_for_lstm = data_preparation_worker.generate_data_for_lstm(
                nr_test_weeks=config["prepare_data"]["test_weeks"],
                lstm_window=lstm_window,
                drop_weeks_before_first_action=config["lstm"][
                    "drop_weeks_before_first_action"
                ],
                churn_after_n_inactive_weeks=config["lstm"][
                    "churn_after_n_inactive_weeks"
                ],
                _save_data=False,
            )

            model = LstmClvModel(paths=self.paths, data=data_for_lstm)
            history = model.train(
                model_type=config["lstm"]["model_type"],
                lstm_units=lstm_units,
                validation_ratio=config["prepare_data"]["validation_ratio"],
                learning_rate=learning_rate,
                nr_epochs=config["training"]["epochs"],
                batch_size=batch_size,
                shuffle=config["training"]["shuffle"],
                early_stop_after_epochs=config["training"]["early_stop_after_epochs"],
                verbose=0,
                _save_model=False,
            )
            result: float = round(history.history[target][-1], 4)
            try:
                with open(self.paths.parameter_grid, "a") as log:
                    log.write(f"{result}: {parms}\n")
            except:
                pass
            parms["loss"] = result
            return (result, parms)

        logger.info(
            f"Computing {max_combinations} out of {len(self._grid)} grid combinations "
            f"using {cpu_count()} available CPUs."
        )

        grid = random.sample(self._grid, max_combinations)
        pool = Pool()

        results: List[Tuple[float, Dict[str, Any]]] = pool.map(train_lstm, grid)  # type: ignore

        self.results = sorted(results, key=lambda x: x[0])

        best_parms = self.results[0][1]
        self._update_config(**best_parms)  # type: ignore
        logger.info(f"Config updated with best parameters: {best_parms}")

        return pd.DataFrame([r[1] for r in results])  # returns tidy output

    def _update_config(
        self,
        lstm_window: int,
        lstm_units: int,
        learning_rate: float,
        batch_size: int,
        **kwargs,
    ) -> None:
        self._config["lstm"]["window"] = lstm_window
        self._config["lstm"]["lstm_units"] = lstm_units
        self._config["training"]["learning_rate"] = learning_rate
        self._config["training"]["batch_size"] = batch_size
        self._config.save()

    @staticmethod
    def _permutate_dict_of_lists(dict_of_lists: Parms) -> List[Parms]:
        # https://stackoverflow.com/a/61335465
        keys, values = zip(*dict_of_lists.items())
        out = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return out  # type: ignore
