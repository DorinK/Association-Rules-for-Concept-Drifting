import logging
from typing import Tuple, Any, List
import pandas as pd
from efficient_apriori import apriori

from association_finder.cutoff_values_finder import CutoffValuesFinder
from association_finder.models import ConceptDriftResult, AssociationRule, Transaction, CutoffValuesType


class ConceptDriftsFinder:
    """
    This class encapsulates the concepts drift finding logic
    """

    def find_concept_drifts(self, transactions: List[Transaction], concept_column: str, target_column: str,
                            min_confidence: float = 0.4, min_support: float = 0.4,
                            diff_threshold: float = 0.1) -> List[ConceptDriftResult]:
        """
        Given a dataset of transactions:
        1. find different <concept_column> cutoffs
        2. split dataset based on the cutoff
        3. find rules for each subset of the dataset
        4. find rules that are new given a specific subset
        """

        concept_drift_results = []

        # Choose cutoff values
        cutoff_values, cutoff_values_type = CutoffValuesFinder.choose_cutoff_values(transactions, concept_column)

        logging.debug(f"Found concept_column: {concept_column} ; cutoff_values_type: {cutoff_values_type} ; len(cutoff_values) {len(cutoff_values)}")

        # Run over cutoff values
        for concept_cutoff in cutoff_values:
            # Split dataset
            part_one, part_two = self._split_dataset(transactions, concept_column, concept_cutoff=concept_cutoff,
                                                     cutoff_values_type=cutoff_values_type)

            # Skip if any of the parts are empty
            if not any(part_one) or not any(part_two):
                continue

            logging.debug(f"Start concept_cutoff: {concept_cutoff}")

            # Calc rules for each split
            rules_one = self._calc_rules(part_one, target_column)
            rules_two = self._calc_rules(part_two, target_column)

            logging.debug(f"Start comparing rules: {concept_cutoff}")

            # Compare rules and find concept drifts
            concept_drift_results.extend(self._compare_rules(rules_one, rules_two, min_confidence, min_support,
                                                             concept_cutoff, concept_column, diff_threshold))
        return concept_drift_results

    def _split_dataset(self, transactions: List[Transaction], concept_column: str, concept_cutoff: Any,
                       cutoff_values_type: CutoffValuesType) -> Tuple[list, list]:
        """
        Split the dataset into two based on a given cutoff.
        Discrete type: Choose only values that are equal / unequal to concept cutoff.
        Numeric type: Choose values that are higher / lower than concept cutoff.
        """

        if cutoff_values_type == CutoffValuesType.Discrete:
            transactions_one = [transaction for transaction in transactions if
                                transaction.items[concept_column] != concept_cutoff]
            transactions_two = [transaction for transaction in transactions if
                                transaction.items[concept_column] == concept_cutoff]
        else:
            transactions_one = [transaction for transaction in transactions if
                                transaction.items[concept_column] < concept_cutoff]
            transactions_two = [transaction for transaction in transactions if
                                transaction.items[concept_column] >= concept_cutoff]

        return transactions_one, transactions_two

    def _calc_rules(self, transactions: List[Transaction], target_column: str) -> List[AssociationRule]:
        """
        Run the apriori algorithm for the subset of the dataset
        """

        # Prepare the data for the apriori algorithm
        apriori_input: List[List[Tuple[str, str]]] = self._convert_transactions_to_apriori_input(transactions)

        logging.debug("Starting apriori")

        # Note: the reason we use `min_confidence=0.1` and `min_support=0.1` is that we need all rules for comparison.
        # The confidence / support filters will run afterwards
        itemsets, rules = apriori(apriori_input, min_confidence=0.1, min_support=0.1)

        logging.debug("Finished apriori")

        # Convert rules to our own objects (used to easily filter for target_column)
        rules_parsed = [AssociationRule.create(rule) for rule in rules]

        # Keep only rules that have only the <target_column> in their right hand size
        # TODO: Consider if we want rhs to be exactly 1
        target_rules = [rule for rule in rules_parsed if
                        target_column in rule.right_hand_side and len(rule.right_hand_side) == 1]

        return target_rules

    def _convert_transactions_to_apriori_input(self, transactions: List[Transaction]) -> List[List[Tuple[str, str]]]:
        """
        The input for the apriori algorithm is a list of transactions (e.g., [('eggs', 'milk'), ('eggs', ...), ...]
        """

        return [[(item_key, item_value) for item_key, item_value in transaction.items.items()] for transaction in
                transactions]

    def _compare_rules(self, rules_one: List[AssociationRule], rules_two: List[AssociationRule],
                       min_confidence: float, min_support: float, concept_cutoff: float,
                       concept_column: str, diff_threshold: float) -> List[ConceptDriftResult]:
        # Create a unique list of all pairs of rules by iterating over the two lists
        rules_pairs = []
        for rule_one in rules_one:
            rule_two = find_equivalent_rule(rule_one, rules_two)
            rules_pairs.append((rule_one, rule_two))

        for rule_two in rules_two:
            rule_one = find_equivalent_rule(rule_two, rules_one)
            if (rule_one, rule_two) not in rules_pairs:
                rules_pairs.append((rule_one, rule_two))

        # Find Concept drifts
        concept_drifts = []
        for rules in rules_pairs:
            # Here is where we actually filter the rules, keep only pairs where one of them is valid.
            any_rule_valid = any(rule for rule in rules if is_rule_valid(rule, min_confidence, min_support))

            if any_rule_valid:
                confidence_before = rules[0].confidence if rules[0] else None
                confidence_after = rules[1].confidence if rules[1] else None
                support_before = rules[0].support if rules[0] else None
                support_after = rules[1].support if rules[1] else None
                lift_before = rules[0].lift if rules[0] else None
                lift_after = rules[1].lift if rules[1] else None

                # Remove rules that have the same confidence as they are not interesting
                are_rules_different = confidence_before != confidence_after
                # Remove pairs that don't pass the diff threshold
                is_diff_threshold_valid = diff_threshold is None or (confidence_before and confidence_after and abs(
                    confidence_before - confidence_after) > diff_threshold)
                if are_rules_different and is_diff_threshold_valid:
                    # Save any of the two rules that is not None
                    non_none_rule = [rule for rule in rules if rule is not None][0]
                    concept_drifts.append(ConceptDriftResult(
                        non_none_rule.left_hand_side,
                        non_none_rule.right_hand_side,
                        confidence_before,
                        confidence_after,
                        support_before,
                        support_after,
                        lift_before,
                        lift_after,
                        concept_cutoff,
                        concept_column
                    ))

        return concept_drifts


def find_equivalent_rule(rule_to_find: AssociationRule, rules: List[AssociationRule]):
    """
    Given a specific rule that came from one part of the dataset, find the equivalent rule in the list of rules that
    came from the other part of the dataset.
    """

    found_rule = None
    for rule in rules:
        # If both rules match - we found the rule
        if rule.right_hand_side == rule_to_find.right_hand_side and rule.left_hand_side == rule_to_find.left_hand_side:
            found_rule = rule
            break

    return found_rule


def is_rule_valid(rule: AssociationRule, min_confidence: float, min_support: float):
    """
    This is where we actually filter the rules. A valid rule needs to pass the confidence / support thresholds.
    """
    return rule is not None and rule.confidence >= min_confidence and rule.support >= min_support


def convert_df_to_transactions(df: pd.DataFrame) -> List[Transaction]:
    records = df.to_dict(orient='records')
    transactions = []
    for r in records:
        transactions.append(Transaction({k: v for k, v in r.items()}))

    return transactions
