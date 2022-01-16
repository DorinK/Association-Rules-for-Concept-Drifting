from dataclasses import dataclass
from typing import Any, List, Dict

import efficient_apriori

SEPARATOR = ":"


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
        left_hand_side_rules = {key_value_str.split(SEPARATOR)[0]: key_value_str.split(SEPARATOR)[1] for key_value_str in rule.lhs}
        right_hand_side_rules = {key_value_str.split(SEPARATOR)[0]: key_value_str.split(SEPARATOR)[1] for key_value_str in rule.rhs}
        return cls(left_hand_side_rules, right_hand_side_rules, rule.confidence, rule.support, rule.lift, rule.conviction)


@dataclass
class ConceptDriftResult:
    rule: AssociationRule
    confidence_before: float
    confidence_after: float
    concept_cutoff: Any

    def confidence_diff(self):
        if self.confidence_before and self.confidence_after:
            return abs(self.confidence_before - self.confidence_after)
        else:
            return None
