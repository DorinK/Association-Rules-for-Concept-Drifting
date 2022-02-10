from dataclasses import dataclass
from enum import Enum
from typing import List

import pytest
from dataclasses_json import dataclass_json

from association_finder.models import Transaction


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
    sales.append(MockHousingSale(sales_price=1, sales_year=2015, construction_year=1991, has_ac=True, num_rooms=3))
    sales.append(MockHousingSale(sales_price=1, sales_year=2015, construction_year=1992, has_ac=True, num_rooms=3))

    # In 2020, AC did influence price (same house is more expensive)
    sales.append(MockHousingSale(sales_price=1, sales_year=2020, construction_year=1993, has_ac=False, num_rooms=3))
    sales.append(MockHousingSale(sales_price=2, sales_year=2020, construction_year=1994, has_ac=True, num_rooms=3))
    sales.append(MockHousingSale(sales_price=2, sales_year=2020, construction_year=1995, has_ac=True, num_rooms=3))

    # Convert to transactions objects (this is the input to ConceptDriftsFinder)
    return [Transaction(transaction.to_dict()) for transaction in sales]
