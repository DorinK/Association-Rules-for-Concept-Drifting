import logging
from typing import List

import pandas as pd
from math import ceil, floor

from association_finder.concept_drifts_finder import ConceptDriftsFinder
from association_finder.models import Transaction, ConceptDriftResult

CONCEPT_DRIFT_SIZE_THRESHOLD = 50


class ConceptEngineering:
    """
    Feature engineering based on concept drifting
    """

    def __init__(self, min_confidence=0.4, min_support=0.4, diff_threshold=0.1, concept_drift_size_threshold=50,
                 verbose=False):
        self.diff_threshold = diff_threshold
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.concept_drift_size_threshold = concept_drift_size_threshold
        self.concepts_to_skip = []
        self.failed_concepts = []
        self.concepts_df = None
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, df: pd.DataFrame, target_column: str, one_hot_columns: List[str]):
        """
        Finds the concept values, and the feature changes we want to make to the dataframe
        """

        self.one_hot_columns = one_hot_columns
        self._build_concepts_df(df, target_column)
        X_rules = X.copy()
        self._modify_X(X_rules, should_update_concepts_to_skip=True)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the feature changes found in the fit method over the dataframe
        """

        X_rules = X.copy()
        self._modify_X(X_rules)
        return X_rules

    def fit_transform(self, X: pd.DataFrame, df: pd.DataFrame, target_column: str, one_hot_columns: List[str]):
        self.fit(X, df, target_column, one_hot_columns)
        return self.transform(X)

    def _build_concepts_df(self, df: pd.DataFrame, target_column):
        # We need to convert our dataframe to a list of transactions
        records = df.to_dict(orient='records')
        transactions = []
        for r in records:
            transactions.append(Transaction({k: v for k, v in r.items()}))

        potential_concept_columns = list(df.columns)
        potential_concept_columns.remove(target_column)
        all_concepts = []
        for concept_column in potential_concept_columns:
            try:
                # Run the ConceptDriftsFinder
                concepts: List[ConceptDriftResult] = ConceptDriftsFinder().find_concept_drifts(transactions,
                                                                                               concept_column,
                                                                                               target_column,
                                                                                               min_confidence=self.min_confidence,
                                                                                               min_support=self.min_support,
                                                                                               diff_threshold=self.diff_threshold)

                # Convert to dataframe
                all_concepts.extend(concepts)
            except:
                if self.verbose:
                    logging.exception(f"Failed concept column '{concept_column}'")
                self.failed_concepts.append(concept_column)

        concepts_df = pd.DataFrame([x.to_dict() for x in all_concepts])
        self.concepts_df = concepts_df

    def _modify_X(self, X, should_update_concepts_to_skip = False):
        for idx, concept_row in self.concepts_df.iterrows():
            if idx in self.concepts_to_skip:
                # print(f"Skip concept {idx}")
                continue

            # print()
            # print()
            if concept_row['concept_column'] != 'OverallQual':
                continue

            filtered_X = self._filter_X_by_lhs(X, concept_row)
            filtered_X = self._filter_X_by_concept(filtered_X, concept_row)

            # Skip too small concept drifts
            if should_update_concepts_to_skip and filtered_X.shape[0] < self.concept_drift_size_threshold:
                self.concepts_to_skip.append(idx)
                # print("Skip")
                continue

            lhs_columns = []
            for column_name, column_value in concept_row['left_hand_side'].items():
                if column_name not in self.one_hot_columns:
                    lhs_columns.append(column_name)
                else:
                    lhs_columns.append(f"{column_name}_{column_value}")

            # print(concept_row)
            X.loc[filtered_X.index, lhs_columns] = X.loc[filtered_X.index, lhs_columns] * self._get_weight(concept_row)

        return X

    def _get_weight(self, concept_row) -> float:
        """
        Decides how much to increase confident rules
        """

        return 2.0
        # TODO: Try concept_row confidence diff

    def _from_lhs_to_queries(self, lhs: dict) -> List[str]:
        """
        Converts left_hand_side dictionary to pandas queries
        """

        queries = []
        for column_name, column_value in lhs.items():
            if column_name not in self.one_hot_columns:
                # TODO: Consider to use for continuous values < > instead of exact match
                queries.append(f"{column_name} == {column_value}")
            else:
                queries.append(f"{column_name}_{column_value} == 1")
        return queries

    def _filter_X_by_lhs(self, X: pd.DataFrame, concept_row) -> pd.DataFrame:
        lhs_queries = self._from_lhs_to_queries(concept_row['left_hand_side'])

        filtered_X = X
        for lhs_query in lhs_queries:
            filtered_X = filtered_X.query(lhs_query)

        return filtered_X

    def _create_concept_query(self, X, concept_row) -> str:
        concept_column = concept_row['concept_column']
        concept_cutoff = concept_row['concept_cutoff']
        is_categorical_column = str(X[concept_column].dtype) == "category"

        # Increase those that are lower than the concept
        if concept_row['confidence_before'] >= concept_row['confidence_after']:
            # Example: If we got 4.2 in the qcut, we need to change it to 5
            if is_categorical_column:
                concept_cutoff = ceil(concept_cutoff)

            concept_query = f"{concept_column} < {concept_cutoff}"
        # Increase those that are bigger than the concept
        else:
            # Example: If we got 4.2 in the qcut, we need to change it to 4
            if is_categorical_column:
                concept_cutoff = floor(concept_cutoff)

            concept_query = f"{concept_column} >= {concept_cutoff}"

        return concept_query

    def _filter_X_by_concept(self, X: pd.DataFrame, concept_row) -> pd.DataFrame:
        concept_query = self._create_concept_query(X, concept_row)
        return X.query(concept_query)


