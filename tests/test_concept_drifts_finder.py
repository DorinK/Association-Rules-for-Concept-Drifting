from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pytest
from dataclasses_json import dataclass_json

from association_finder.concept_drifts_finder import ConceptDriftsFinder, find_equivalent_rule
from association_finder.models import ConceptDriftResult, AssociationRule, Transaction


class SalesPrice(Enum):
    """
    A housing dataset sales price buckets used for tests
    """

    Low = 0
    Medium = 1
    High = 2


@dataclass_json
@dataclass
class MockHousingSale:
    """
    A housing dataset object used for tests
    """

    construction_year: int
    sales_year: int
    sales_price: SalesPrice.High.value
    has_ac: bool
    num_rooms: int


@pytest.fixture
def sales_data_fixture__ac_influences_price_over_time() -> List[Transaction]:
    """
    This mock dataframe simulates a case where prices went up following AC demand
    """

    sales = []

    # In 2015, AC didn't influence price
    sales.append(MockHousingSale(sales_price=1, sales_year=2015, construction_year=1990, has_ac=False, num_rooms=3))
    sales.append(MockHousingSale(sales_price=1, sales_year=2015, construction_year=2010, has_ac=True, num_rooms=3))

    # In 2020, AC did influence price (same house is more expensive)
    sales.append(MockHousingSale(sales_price=1, sales_year=2020, construction_year=1990, has_ac=False, num_rooms=3))
    sales.append(MockHousingSale(sales_price=2, sales_year=2020, construction_year=2010, has_ac=True, num_rooms=3))

    # Convert to transactions objects (this is the input to ConceptDriftsFinder)
    return [Transaction(transaction.to_dict()) for transaction in sales]


class TestAssociationFinder:
    def test_association_finder(self, sales_data_fixture__ac_influences_price_over_time):
        """
        In this test, we expect to see the rule (has_ac: True) -> (sales_price: 2)
        """

        # Find concept drifts
        concept_drift_results: List[ConceptDriftResult] = ConceptDriftsFinder().find_concept_drifts(
            sales_data_fixture__ac_influences_price_over_time, concept_column="sales_year", target_column="sales_price")

        # We expect to find the rule (has_ac) -> (sales_price: 2)
        concept = find_equivalent_concept(AssociationRule({"has_ac": True}, {"sales_price": 2}, None, None, None, None), concept_drift_results)
        assert concept is not None
        assert concept.confidence_before is None
        assert concept.confidence_after == 1.0


def find_equivalent_concept(rule_to_find: AssociationRule, concepts: List[ConceptDriftResult]) -> Optional[ConceptDriftResult]:
    """
    Helper function to find a specific rule in a list of concepts
    """

    found_concept = None
    for concept in concepts:
        # If both rules match - we found the concept
        if concept.rule.right_hand_side == rule_to_find.right_hand_side and concept.rule.left_hand_side == rule_to_find.left_hand_side:
            found_concept = concept
            break

    return found_concept
