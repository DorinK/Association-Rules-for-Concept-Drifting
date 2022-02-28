import pytest

from association_finder.concept_drifts_finder import ConceptDriftsFinder
from association_finder.concept_engineering import ConceptEngineering
import pandas as pd
import mock

from association_finder.models import ConceptDriftResult


class TestConceptEngineering:

    @pytest.fixture
    def df(self) -> pd.DataFrame:
        rows = [
            (3, 1, 1),
        ]

        return pd.DataFrame(rows, columns=["Humidity9am", "RainToday", "RainTomorrow"])

    def test_fit_transform__test_bug_where_one_rule_causes_the_next_not_to_update(self, df):
        """
        This test is used to check the bug where one rule causes the next not to update.
        For example, if two rules look for "Humidity9am": 3.
        The first one might multiply it by 2 to 6, but then the second rule won't find the value 3, so it will stay 6.

        Similarly, another rule that looks at the value 6 might run on something that it shouldn't run.
        """

        target_column = "RainTomorrow"
        concept_engineering = ConceptEngineering(concept_drift_size_threshold=1)
        X = df.drop(columns=[target_column])

        with mock.patch.object(ConceptDriftsFinder, "find_concept_drifts") as find_concept_drifts_mock:
            find_concept_drifts_mock.return_value = [
                ConceptDriftResult({"Humidity9am": 3}, {"RainTomorrow": 1}, 0.4, 0.6, 0.4, 0.6, 0.0, "RainToday")
            ]
            new_X = concept_engineering.fit_transform(X, df, target_column, [])

        # We expect the new value to be 12 (twice multiplied by 2)
        assert new_X.iloc[0]['Humidity9am'] == 12
