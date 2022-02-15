from typing import List, Tuple, Any
import pandas as pd
from association_finder.models import Transaction, CutoffValuesType


class CutoffValuesFinder:
    """
    This class is in charge of finding cutoff values
    """

    @staticmethod
    def choose_cutoff_values(transactions: List[Transaction],
                             concept_column: str) -> Tuple[List[Any], CutoffValuesType]:
        # Find all unique values for target columns
        unique_target_values = list(set(
            [transaction.items[concept_column] for transaction in transactions if concept_column in transaction.items]))

        cutoff_values_type = CutoffValuesFinder._choose_cutoff_value_type(unique_target_values)
        if cutoff_values_type == CutoffValuesType.Discrete:
            cutoff_values = CutoffValuesFinder._find_discrete_cutoff_values(unique_target_values)
        else:
            cutoff_values = CutoffValuesFinder._find_continuous_cutoff_values(unique_target_values)

        # Return all values
        return cutoff_values, cutoff_values_type

    @staticmethod
    def _choose_cutoff_value_type(unique_target_values: List[Any]) -> CutoffValuesType:
        # If not all values are numeric, return discrete
        is_numeric = lambda x: isinstance(x, int) or isinstance(x, float)
        all_values_are_numeric = all(value for value in unique_target_values if is_numeric(value))
        if not all_values_are_numeric:
            return CutoffValuesType.Discrete

        # If more than 5 values, return continuous. Otherwise discrete
        if len(unique_target_values) < 5:
            return CutoffValuesType.Discrete
        else:
            return CutoffValuesType.Continuous

    @staticmethod
    def _find_discrete_cutoff_values(unique_target_values: List[Any]) -> List[Any]:
        # In discrete columns we want to choose all options as cutoff
        return unique_target_values

    @staticmethod
    def _find_continuous_cutoff_values(unique_target_values: List[Any]) -> List[Any]:
        try:
            # Bin numerical data using quantiles
            intervals = pd.qcut(unique_target_values, 5).categories
        except:
            # sometimes for highly skewed data, we cannot perform qcut as most quantiles are equal
            intervals = pd.cut(unique_target_values, 5).categories

        # Take either side of the interval. For example, from (1989.999, 1991.0) take 1989.999
        return [interval.left for interval in intervals]
