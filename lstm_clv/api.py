from lstm_clv.libs import Config, ExperimentPaths, join_paths
from lstm_clv.pipeline.benchmark import ParetoNBDGammaGammaWorker
from lstm_clv.pipeline.lstm_model import ClvPredictions, History, LstmClvModel
from lstm_clv.pipeline.preparation import DataPreparationWorker


def create_experiment(config: Config) -> None:
    """Create folder structure for a new experiment."""
    experiment_name = config["experiment"]
    paths = ExperimentPaths(experiment_name)
    paths.initiate_folders()
    config.save(paths.config)


def prepare_data(experiment: str) -> DataPreparationWorker:
    """Prepare data for modeling by transforming, filtering, and scaling it."""

    paths = ExperimentPaths(experiment)
    config = Config.load(paths.config)
    worker = DataPreparationWorker(paths)

    raw_data_path = config["prepare_data"]["raw_data_path"]

    worker.load_data(
        transactions_path=join_paths(raw_data_path, "transactions.csv"),
        attributes_path=join_paths(raw_data_path, "user_attributes.csv"),
        margins_path=join_paths(raw_data_path, "margins.json"),
    )
    worker.prepare_data(
        aggregate_events=config["prepare_data"]["aggregate_events"],
        dummify_events=config["prepare_data"]["dummify_events"],
        nr_test_weeks=config["prepare_data"]["test_weeks"],
        min_nr_actions=config["prepare_data"]["min_nr_actions"],
        keep_user_ids=config["prepare_data"]["keep_user_ids"],
    )
    worker.scale_data(
        transformation=config["scaling"]["transformation"],
        outlier_threshold=config["scaling"]["outlier_threshold"],
    )
    worker.generate_data_for_lstm(
        nr_test_weeks=config["prepare_data"]["test_weeks"],
        lstm_window=config["lstm"]["window"],
        drop_weeks_before_first_action=config["lstm"]["drop_weeks_before_first_action"],
        churn_after_n_inactive_weeks=config["lstm"]["churn_after_n_inactive_weeks"],
    )
    return worker


def evaluate_benchmark(experiment: str) -> ParetoNBDGammaGammaWorker:
    """Fit and evaluate benchmark model (Pareto/NBD with a Gamma-Gamma submodel)."""

    paths = ExperimentPaths(experiment)
    config = Config.load(paths.config)
    worker = ParetoNBDGammaGammaWorker(paths)

    nr_test_weeks = config["prepare_data"]["test_weeks"]

    worker.fit(nr_test_weeks=nr_test_weeks, penalizer_coef=0.01)
    worker.predict_and_evaluate(nr_test_weeks=nr_test_weeks, discount_rate=0.0)
    return worker


def train_lstm(experiment: str) -> History:
    """doc"""

    paths = ExperimentPaths(experiment)
    config = Config.load(paths.config)
    model = LstmClvModel(paths)

    history = model.train(
        model_type=config["lstm"]["model_type"],
        lstm_units=config["lstm"]["lstm_units"],
        validation_ratio=config["prepare_data"]["validation_ratio"],
        learning_rate=config["training"]["learning_rate"],
        nr_epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"]["shuffle"],
        early_stop_after_epochs=config["training"]["early_stop_after_epochs"],
        verbose=config["training"]["verbose"],
    )
    return history


def evaluate_lstm(experiment: str) -> ClvPredictions:
    """doc"""

    paths = ExperimentPaths(experiment)
    config = Config.load(paths.config)
    model = LstmClvModel(paths)

    results = model.predict_and_evaluate(nr_weeks=config["prepare_data"]["test_weeks"])
    return results
