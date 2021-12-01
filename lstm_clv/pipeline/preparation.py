from typing import List, Optional

import pandas as pd
from lstm_clv.data_types import (
    DataForLstm,
    EventsMetadata,
    Margins,
    Scaler,
    UserAttributes,
    UserEvents,
)
from lstm_clv.libs import ExperimentPaths, get_logger, pandas_utils

logger = get_logger("preparation")


class DataPreparationWorker:
    transactions_raw: pd.DataFrame
    transactions_train: UserEvents
    attributes: UserAttributes
    margins: Margins
    transactions_scaler: Scaler
    attributes_scaler: Scaler
    user_ids: List[str]

    def __init__(self, paths: ExperimentPaths) -> None:
        self.paths = paths

    def load_data(
        self, transactions_path: str, attributes_path: str, margins_path: str
    ) -> None:
        """
        Step 1. Load source data
        """
        self.transactions_raw = pandas_utils.read_events_df(transactions_path)

        try:
            self.attributes = UserAttributes.load(attributes_path)
        except FileNotFoundError:
            logger.warning("User attributes not available, using 1 for all users.")
            self.attributes = UserAttributes()

        try:
            self.margins = Margins.load(margins_path)
        except FileNotFoundError:
            logger.warning("Profit margins not available, using 1 for all events.")
            events = EventsMetadata(self.transactions_raw).events
            self.margins = Margins({e: 1 for e in events})

    def prepare_data(
        self,
        aggregate_events: bool,
        dummify_events: bool,
        nr_test_weeks: int,
        min_nr_actions: int,
        keep_user_ids: Optional[List[str]],
    ) -> None:
        """
        Step 2. Prepare source data for modeling
        """
        # prepare a matrix with transactional data
        transactions = UserEvents.construct(
            df=self.transactions_raw,
            aggregate_events=aggregate_events,
            dummify_events=dummify_events,
            profit_margins=self.margins,
        )
        last_week_train = transactions.nr_weeks - nr_test_weeks

        # remove users with insufficient number of txs in train timeframe
        if keep_user_ids:
            logger.info("Keeping only users from the provided list")
            all_user_ids = transactions.get_user_ids()
            keep_user_indices = list()
            for user_id in keep_user_ids:
                index = all_user_ids.index(user_id)
                keep_user_indices.append(index)
        else:
            logger.info(f"Keeping only users with > {min_nr_actions} actions")
            _txs_train = transactions.filter(
                week_to=last_week_train
            ).nr_transactions_per_user
            keep_user_indices = list((_txs_train > min_nr_actions).nonzero()[0])  # type: ignore

        transactions = transactions.filter(user_indices=keep_user_indices)
        self.transactions_train = transactions.filter(week_to=last_week_train)
        self.user_ids = transactions.get_user_ids()

        logger.info(f"All data: {transactions}")
        logger.info(f"Train: {self.transactions_train}")

        # drop redundant users from attribute data
        self.attributes = self.attributes.filter(user_ids=transactions.get_user_ids())

        # save data
        transactions.save(self.paths.transactions_matrix)
        self.attributes.save(self.paths.user_attrs)
        self.margins.save(self.paths.margins)

    def scale_data(self, transformation: str, outlier_threshold: int) -> None:
        """
        Step 3. Fit data scalers based on train data
        """
        self.transactions_scaler = Scaler.fit_events_data(
            self.transactions_train,
            transformation=transformation,  # type: ignore
            outlier_threshold=outlier_threshold,
        )
        self.attributes_scaler = Scaler.fit_user_data(
            self.attributes, outlier_threshold=outlier_threshold
        )
        self.transactions_scaler.save(self.paths.transactions_scaler)
        self.attributes_scaler.save(self.paths.user_attrs_scaler)

    def __reload_data_if_needed(self, nr_test_weeks: int) -> None:
        try:
            self.transactions_scaler  # load data if previous steps not executed
        except AttributeError:
            transactions = UserEvents.load(self.paths.transactions_matrix)
            last_week_train = transactions.nr_weeks - nr_test_weeks
            self.transactions_train = transactions.filter(week_to=last_week_train)

            self.attributes = UserAttributes.load(self.paths.user_attrs)
            self.transactions_scaler = Scaler.load(self.paths.transactions_scaler)
            self.attributes_scaler = Scaler.load(self.paths.user_attrs_scaler)

    def generate_data_for_lstm(
        self,
        nr_test_weeks: int,
        lstm_window: int,
        drop_weeks_before_first_action: bool,
        churn_after_n_inactive_weeks: int,
        _save_data: bool = True,
    ) -> DataForLstm:
        """
        Step 4. Prepare input data for the LSTM-CLV model.
        This method can be used independently of previous steps.
        """
        self.__reload_data_if_needed(nr_test_weeks=nr_test_weeks)

        txs_scaled = self.transactions_scaler.transform(self.transactions_train)
        attrs_scaled = self.attributes_scaler.transform(self.attributes)
        data_for_lstm = DataForLstm.construct(
            transactions=txs_scaled,
            user_attrs=attrs_scaled,
            window=lstm_window,
            drop_weeks_before_first_action=drop_weeks_before_first_action,
            churn_after_n_inactive_weeks=churn_after_n_inactive_weeks,
        )
        if _save_data:
            data_for_lstm.save(self.paths.data_for_lstm)
        return data_for_lstm
