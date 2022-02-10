from typing import List, Optional

from association_finder.concept_drifts_finder import ConceptDriftsFinder, find_equivalent_rule
from association_finder.models import ConceptDriftResult, AssociationRule


class TestAssociationFinder:
    def test_association_finder(self, sales_data_fixture__ac_influences_price_over_time):
        """
        In this test, we expect to see the rule (has_ac: True) -> (sales_price: 2)
        """

        # Find concept drifts
        concept_drift_results: List[ConceptDriftResult] = ConceptDriftsFinder().find_concept_drifts(
            sales_data_fixture__ac_influences_price_over_time, concept_column="sales_year", target_column="sales_price",
            min_confidence=0.5, diff_threshold=None)

        # We expect to find the rule (has_ac) -> (sales_price: 2)
        rule_to_find = AssociationRule({"has_ac": True}, {"sales_price": 2}, None, None, None, None)
        concept = find_equivalent_concept(rule_to_find,
                                          concept_drift_results)
        assert concept is not None
        assert concept.confidence_before is None
        assert concept.confidence_after == 1.0


def find_equivalent_concept(rule_to_find: AssociationRule, concepts: List[ConceptDriftResult]) -> Optional[
    ConceptDriftResult]:
    """
    Helper function to find a specific rule in a list of concepts
    """

    found_concept = None
    for concept in concepts:
        # If both rules match - we found the concept
        if concept.right_hand_side == rule_to_find.right_hand_side and concept.left_hand_side == rule_to_find.left_hand_side:
            found_concept = concept
            break

    return found_concept
