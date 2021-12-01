import pandas as pd
from lifetimes import GammaGammaFitter, ParetoNBDFitter
from lstm_clv.data_types import Margins, UserEvents
from lstm_clv.libs import ExperimentPaths, ModelPerformance, get_logger

logger = get_logger("benchmark")


class ParetoNBDGammaGammaModel:
    """
    Extended Pareto/NBD with Gamma-Gamma submodel as implemented
    in the lifetimes package
    """

    def __init__(self, penalizer_coef: float = 0.1) -> None:
        self._pareto_nbd = ParetoNBDFitter(penalizer_coef=penalizer_coef)
        self._gammagamma = GammaGammaFitter(penalizer_coef=penalizer_coef)

    def fit(self, train_df: pd.DataFrame) -> None:
        """doc"""
        # non-recurring users can't be processed by the model
        # save them for latter model evaluation (predict zeroes for these)
        self.train_df_orig = train_df.copy()
        self.train_df = train_df

        self._pareto_nbd.fit(
            frequency=self.train_df["frequency"],
            recency=self.train_df["recency"],
            T=self.train_df["age"],
        )
        self._gammagamma.fit(
            frequency=self.train_df["frequency"],
            monetary_value=self.train_df["monetary_value_avg"],
        )

    def predict(
        self, test_df: pd.DataFrame, nr_weeks: int, discount_rate=0
    ) -> pd.DataFrame:
        """doc"""
        preds = self.train_df[["user_id"]]
        preds["clv_pred"] = self._gammagamma.customer_lifetime_value(
            transaction_prediction_model=self._pareto_nbd,
            frequency=self.train_df["frequency"],
            recency=self.train_df["recency"],
            T=self.train_df["age"],
            monetary_value=self.train_df["monetary_value_avg"],
            time=nr_weeks,
            discount_rate=discount_rate,
            freq="W",
        )

        true = test_df[["user_id"]]
        true["clv_true"] = round(test_df["monetary_value_total"], 2)  # type: ignore

        output = self.train_df_orig.copy()
        output["monetary_value"] = round(output["monetary_value_avg"], 2)  # type: ignore
        output = output.merge(preds, on="user_id", how="left")
        output = output.merge(true, on="user_id", how="left")
        output = output.fillna(0)
        return output


class ParetoNBDGammaGammaWorker:
    def __init__(self, paths: ExperimentPaths):
        self.paths = paths

    def fit(self, nr_test_weeks: int, penalizer_coef: float):

        transactions = UserEvents.load(self.paths.transactions_matrix)
        margins = Margins.load(self.paths.margins)

        first_week_test = transactions.nr_weeks - nr_test_weeks
        train = transactions.filter(week_to=first_week_test)
        test = transactions.filter(week_from=first_week_test)

        logger.info(
            f"Predicting values for weeks {first_week_test}--{transactions.nr_weeks} "
            f"({test.nr_weeks} weeks) for {test.nr_users} users."
        )

        train_df = train.get_summary(profit_margins=margins)
        self.test_df = test.get_summary(profit_margins=margins)

        self.benchmark = ParetoNBDGammaGammaModel(penalizer_coef=penalizer_coef)
        self.benchmark.fit(train_df)

    def predict_and_evaluate(self, nr_test_weeks: int, discount_rate: float):
        self.preds = self.benchmark.predict(
            test_df=self.test_df, nr_weeks=nr_test_weeks, discount_rate=discount_rate
        )
        self.preds.to_csv(self.paths.benchmark_preds, index=False, float_format="%.2f")

        self.performance = ModelPerformance(
            self.preds["clv_true"].values, self.preds["clv_pred"].values  # type: ignore
        )
        self.performance.save(self.paths.benchmark_performance)
