import logging
import time
from collections import defaultdict
from typing import List

import pandas as pd
from math import ceil, floor

from association_finder.concept_drifts_finder import ConceptDriftsFinder
from association_finder.models import Transaction, ConceptDriftResult

CONCEPT_DRIFT_SIZE_THRESHOLD = 50


class ConceptEngineering:
    """
    Utilize the concept drifting functionality for automatic feature engineering
    """

    def __init__(self, min_confidence=0.4, min_support=0.4, diff_threshold=0.1, concept_drift_size_threshold=50,
                 verbose=False):
        """
        :param min_confidence: The drift is filtered out if the rule in both sides is below min_confidence
        :param min_support: The drift is filtered out if the rule in both sides is below min_support
        :param diff_threshold: The drift is filtered out if the diff in confidence before and after is below diff_threshold
        :param concept_drift_size_threshold: The drift is filtered if either before or after is below 50
        :param verbose: The process can take time, so this will print more information (e..g, progress ; exceptions)
        """
        self.diff_threshold = diff_threshold
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.concept_drift_size_threshold = concept_drift_size_threshold
        self.concepts_to_skip = []
        self.failed_concepts = []
        self.concepts_df = None
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, df: pd.DataFrame, target_column: str, one_hot_columns: List[str],
            filter_concepts_by_target=None):
        """
        Finds the concept values, and the feature changes we want to make to the dataframe
        """

        self.one_hot_columns = one_hot_columns
        self._build_concepts_df(df, target_column)
        X_rules = X.copy()
        self._modify_X(X_rules, filter_concepts_by_target=filter_concepts_by_target,
                       should_update_concepts_to_skip=True)

        return self

    def transform(self, X: pd.DataFrame, filter_concepts_by_target=None, target_column=None) -> pd.DataFrame:
        """
        Apply the feature changes found in the fit method over the dataframe
        """

        X_rules = X.copy()
        X_rules = self._modify_X(X_rules, filter_concepts_by_target=filter_concepts_by_target,
                                 target_column=target_column)
        return X_rules

    def fit_transform(self, X: pd.DataFrame, df: pd.DataFrame, target_column: str, one_hot_columns: List[str],
                      filter_concepts_by_target=None):
        self.fit(X, df, target_column, one_hot_columns, filter_concepts_by_target)
        return self.transform(X)

    def _build_concepts_df(self, df: pd.DataFrame, target_column):
        """
        Builds the dataframe of concepts using the ConceptsDriftsFinder
        """

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
                start_time = time.time()
                if self.verbose:
                    logging.info(f"Starting concept column '{concept_column}'")

                # Run the ConceptDriftsFinder
                concepts: List[ConceptDriftResult] = ConceptDriftsFinder().find_concept_drifts(transactions,
                                                                                               concept_column,
                                                                                               target_column,
                                                                                               min_confidence=self.min_confidence,
                                                                                               min_support=self.min_support,
                                                                                               diff_threshold=self.diff_threshold,
                                                                                               verbose=self.verbose)

                # Convert to dataframe
                all_concepts.extend(concepts)
            except:
                # This will also write the stack trace
                if self.verbose:
                    logging.exception(f"Failed concept column '{concept_column}'")
                else:
                    logging.warning(f"Failed concept column '{concept_column}'")
                self.failed_concepts.append(concept_column)
            finally:
                end_time = time.time()

            if self.verbose:
                time_took = end_time - start_time
                logging.info(f"Finished concept column ; time_took (in seconds): {time_took}")

        concepts_df = pd.DataFrame([x.to_dict() for x in all_concepts])
        self.concepts_df = concepts_df

    def _modify_X(self, X, should_update_concepts_to_skip=False, filter_concepts_by_target=None, target_column=None):
        concepts_df = self.concepts_df

        # Support applying only specific concepts (for one-vs-all use case)
        if filter_concepts_by_target is not None:
            concepts_df = concepts_df[concepts_df['right_hand_side'] == {target_column: filter_concepts_by_target}]

        # Run over all concepts and find the relevant rows to update
        rows_to_update = defaultdict(lambda: defaultdict(list))
        for idx, concept_row in concepts_df.iterrows():
            if idx in self.concepts_to_skip:
                continue

            filtered_X = self._filter_X_by_lhs(X, concept_row)
            filtered_X = self._filter_X_by_concept(filtered_X, concept_row)

            # Skip too small concept drifts
            if should_update_concepts_to_skip and filtered_X.shape[0] < self.concept_drift_size_threshold:
                self.concepts_to_skip.append(idx)
                continue

            lhs_columns = []
            for column_name, column_value in concept_row['left_hand_side'].items():
                if column_name not in self.one_hot_columns:
                    lhs_columns.append(column_name)
                else:
                    lhs_columns.append(f"{column_name}_{column_value}")

            for row_index in filtered_X.index:
                for lhs_column in lhs_columns:
                    rows_to_update[row_index][lhs_column].append(concept_row)

        # Convert categorical columns to float (otherwise we can't multiply their value)
        X = X.astype('float')

        for row_index, columns in rows_to_update.items():
            for column_name, concepts_rows in columns.items():
                try:
                    X.loc[row_index, column_name] = X.loc[row_index, column_name] * self._get_weight(concepts_rows)
                except:
                    # This will also write the stack trace
                    if self.verbose:
                        logging.exception(f"Failed concept row '{row_index}' ; {column_name}")
                    else:
                        logging.warning(f"Failed concept row '{row_index}' ; {column_name}")

        return X

    def _get_weight(self, concepts_rows) -> float:
        """
        Decides how much to increase confident rules
        """

        return max([abs(concept_row['lift_before'] - concept_row['lift_after']) for concept_row in concepts_rows])

    def _from_lhs_to_queries(self, lhs: dict) -> List[str]:
        """
        Converts left_hand_side dictionary to pandas queries
        """

        queries = []
        for column_name, column_value in lhs.items():
            if column_name not in self.one_hot_columns:
                # TODO: Consider to use for continuous values < > instead of exact match
                queries.append(f"`{column_name}` == {column_value}")
            else:
                queries.append(f"`{column_name}_{column_value}` == 1")
        return queries

    def _filter_X_by_lhs(self, X: pd.DataFrame, concept_row) -> pd.DataFrame:
        lhs_queries = self._from_lhs_to_queries(concept_row['left_hand_side'])

        filtered_X = X
        for lhs_query in lhs_queries:
            filtered_X = filtered_X.query(lhs_query)

        return filtered_X

    def _create_concept_query(self, X, concept_row) -> str:
        """
        Create the query used to filter the X dataframe based on the found concept.
        We always query those with the higher confidence.
        """

        concept_column = concept_row['concept_column']
        concept_cutoff = concept_row['concept_cutoff']

        if concept_column in self.one_hot_columns:
            # When working with one hot columns, the value is part of the column name and the value is either 1 or 0
            concept_column = f"{concept_column}_{concept_cutoff}"
            concept_cutoff = 1

        is_categorical_column = str(X[concept_column].dtype) == "category"

        # Increase those that are lower than the concept
        if concept_row['lift_before'] >= concept_row['lift_after']:
            # Example: If we got 4.2 in the qcut, we need to change it to 5
            if is_categorical_column:
                concept_cutoff = ceil(concept_cutoff)

            concept_query = f"`{concept_column}` < {concept_cutoff}"
        # Increase those that are bigger than the concept
        else:
            # Example: If we got 4.2 in the qcut, we need to change it to 4
            if is_categorical_column:
                concept_cutoff = floor(concept_cutoff)

            concept_query = f"`{concept_column}` >= {concept_cutoff}"

        return concept_query

    def _filter_X_by_concept(self, X: pd.DataFrame, concept_row) -> pd.DataFrame:
        """
        Filters the X dataframe based on the found concept.
        """

        concept_query = self._create_concept_query(X, concept_row)
        return X.query(concept_query)
