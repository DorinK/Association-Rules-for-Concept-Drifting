from typing import Tuple, Any, List

from efficient_apriori import apriori

from association_finder.models import ConceptDriftResult, AssociationRule, Transaction


class ConceptDriftsFinder:
    """
    This class encapsulates the concepts drift finding logic
    """

    def find_concept_drifts(self, transactions: List[Transaction], concept_column: str, target_column: str,
                            min_confidence: float = 1.0, min_support: float = 0.5,
                            diff_threshold: float = 0.5) -> List[ConceptDriftResult]:
        """
        Given a dataset (list of dictionaries)
        1. find different <concept_column> cutoffs
        2. split dataset based on the cutoff
        3. find rules for each subset of the dataset
        4. find rules that are new given a specific subset
        """

        concept_drift_results = []
        # Run over cutoff values
        for concept_cutoff in [3]:
            # Split dataset
            part_one, part_two = self._split_dataset(transactions, concept_column, concept_cutoff=concept_cutoff)

            # Calc rules for each split
            rules_one = self._calc_rules(part_one, target_column)
            rules_two = self._calc_rules(part_two, target_column)

            # Compare rules and find concept drifts
            concept_drift_results.extend(self._compare_rules(rules_one, rules_two, min_confidence, min_support, concept_cutoff))
        return concept_drift_results

    def _split_dataset(self, transactions: List[Transaction], concept_column: str, concept_cutoff: Any) -> Tuple[list, list]:
        """
        Split the dataset into two based on a given cutoff
        """

        transactions_one = [transaction for transaction in transactions if transaction.items[concept_column] < concept_cutoff]
        transactions_two = [transaction for transaction in transactions if transaction.items[concept_column] >= concept_cutoff]

        return transactions_one, transactions_two

    def _calc_rules(self, transactions: List[Transaction], target_column: str) -> List[AssociationRule]:
        """
        Run the apriori algorithm for the subset of the dataset
        """

        # Prepare the data for the apriori algorithm
        apriori_input: List[List[Tuple[str, str]]] = self._convert_transactions_to_apriori_input(transactions)

        # Note: the reason we use `min_confidence=0.1` and `min_support=0.1` is that we need all rules for comparison.
        # The confidence / support filters will run afterwards
        itemsets, rules = apriori(apriori_input, min_confidence=0.1, min_support=0.1)

        # Convert rules to our own objects (used to easily filter for target_column)
        rules_parsed = [AssociationRule.create(rule) for rule in rules]

        # Keep only rules that have only the <target_column> in their right hand size
        target_rules = [rule for rule in rules_parsed if target_column in rule.right_hand_side and len(rule.right_hand_side) == 1]

        return target_rules

    def _convert_transactions_to_apriori_input(self, transactions: List[Transaction]) -> List[List[Tuple[str, str]]]:
        """
        The input for the apriori algorithm is a list of transactions (e.g., [('eggs', 'milk'), ('eggs', ...), ...]
        """

        return [[(item_key, item_value) for item_key, item_value in transaction.items.items()] for transaction in transactions]

    def _compare_rules(self, rules_one: List[AssociationRule], rules_two: List[AssociationRule],
                       min_confidence: float, min_support: float, concept_cutoff: float) -> List[ConceptDriftResult]:
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
                # Remove rules that have the same confidence as they are not interesting
                if confidence_before != confidence_after:
                    # Save any of the two rules that is not None
                    non_none_rule = [rule for rule in rules if rule is not None][0]
                    concept_drifts.append(ConceptDriftResult(
                        non_none_rule.left_hand_side,
                        non_none_rule.right_hand_side,
                        confidence_before,
                        confidence_after,
                        concept_cutoff
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
