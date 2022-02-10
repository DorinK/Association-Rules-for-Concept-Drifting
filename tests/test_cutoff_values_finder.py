from association_finder.cutoff_values_finder import CutoffValuesFinder
from association_finder.models import CutoffValuesType


class TestCutoffValuesFinder:
    def test_choose_cutoff_values__string_column(self, sales_data_fixture__ac_influences_price_over_time):
        cutoff_values, cutoff_values_type = CutoffValuesFinder().choose_cutoff_values(
            sales_data_fixture__ac_influences_price_over_time, "has_ac")

        assert cutoff_values_type == CutoffValuesType.Discrete

    def test_choose_cutoff_values__numeric_discrete_column(self, sales_data_fixture__ac_influences_price_over_time):
        cutoff_values, cutoff_values_type = CutoffValuesFinder().choose_cutoff_values(
            sales_data_fixture__ac_influences_price_over_time, "num_rooms")

        assert cutoff_values_type == CutoffValuesType.Discrete

    def test_choose_cutoff_values__numeric_continuous_column(self, sales_data_fixture__ac_influences_price_over_time):
        cutoff_values, cutoff_values_type = CutoffValuesFinder().choose_cutoff_values(
            sales_data_fixture__ac_influences_price_over_time, "construction_year")

        assert cutoff_values_type == CutoffValuesType.Continuous
