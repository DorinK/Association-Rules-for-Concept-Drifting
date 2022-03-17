from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Dict, Tuple

import efficient_apriori
from dataclasses_json import dataclass_json


@dataclass
class Transaction:
    items: Dict[str, str]


@dataclass
class AssociationRule:
    left_hand_side: Dict[str, Any]
    right_hand_side: Dict[str, Any]
    confidence: float
    support: float
    lift: float
    conviction: float

    @classmethod
    def create(cls, rule: efficient_apriori.Rule):
        left_hand_side_rules = {item[0]: item[1] for item in rule.lhs}
        right_hand_side_rules = {item[0]: item[1] for item in rule.rhs}
        return cls(left_hand_side_rules, right_hand_side_rules, rule.confidence, rule.support, rule.lift, rule.conviction)


@dataclass_json
@dataclass
class ConceptDriftResult:
    left_hand_side: Dict[str, Any]
    right_hand_side: Dict[str, Any]
    confidence_before: float
    confidence_after: float
    support_before: float
    support_after: float
    lift_before: float
    lift_after: float
    concept_cutoff: Any
    concept_column: Any

    def confidence_diff(self):
        if self.confidence_before and self.confidence_after:
            return abs(self.confidence_before - self.confidence_after)
        else:
            return None


class CutoffValuesType(Enum):
    """
    Numeric means that in a column with values [2010, 2020], we might choose 2015.
    Discrete means that in a column with values ["2010", "2020"], we will choose "2010", "2020".
    This influences how we cut the data given this cutoff value.
    """

    Continuous = 0
    Discrete = 1

