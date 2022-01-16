from typing import List

import pandas as pd

from association_finder.concept_drifts_finder import ConceptDriftsFinder
from association_finder.models import Transaction, ConceptDriftResult


def test_housing_dataset():
    dataset_path = "../datasets/houseprices/train.csv"
    dtf = pd.read_csv(dataset_path, index_col='Id')

    # Defining numeric and categorical columns
    numeric_columns = dtf.dtypes[(dtf.dtypes == "float64") | (dtf.dtypes == "int64")].index.tolist()
    very_numerical = [nc for nc in numeric_columns if dtf[nc].nunique() > 20]
    categorical_columns = [c for c in dtf.columns if c not in numeric_columns]
    ordinals = list(set(numeric_columns) - set(very_numerical))

    # Filling Null Values with the column's mean
    na_columns = dtf[very_numerical].isna().sum()
    na_columns = na_columns[na_columns > 0]
    for nc in na_columns.index:
        dtf[nc].fillna(dtf[nc].mean(), inplace=True)

    # Dropping and filling NA values for categorical columns:
    # drop if at least 70% are NA:
    nul_cols = dtf[categorical_columns].isna().sum() / len(dtf)
    drop_us = nul_cols[nul_cols > 0.7]
    dtf = dtf.drop(drop_us.index, axis=1)

    # Fill with a new 'na' category:
    categorical_columns = list(set(categorical_columns) - set(drop_us.index))
    dtf[categorical_columns] = dtf[categorical_columns].fillna('na')

    df = dtf.copy()

    # Bin numerical data
    for c in very_numerical:
        try:
            # df[c] = pd.qcut(dtf[c], 5, labels=["very low", "low", "medium", "high", "very high"])
            df[c] = pd.qcut(dtf[c], 5, labels=[1, 2, 3, 4, 5])
        except:
            # sometimes for highly skewed data, we cannot perform qcut as most quantiles are equal
            # df[c] = pd.cut(dtf[c], 5, labels=["very low", "low", "medium", "high", "very high"])
            df[c] = pd.cut(dtf[c], 5, labels=[1, 2, 3, 4, 5])

    # Focusing on prominent columns:
    good_columns = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'OverallQual', 'OverallCond', 'BldgType', 'LotArea',
                    'GrLivArea', 'FullBath', 'BedroomAbvGr', 'LotFrontage', 'TotalBsmtSF', 'SalePrice']

    # We need to convert our dataframe to a list of transactions
    records = df[good_columns].to_dict(orient='records')
    transactions = []
    for r in records:
        transactions.append(Transaction({k: v for k, v in r.items()}))

    # Run the ConceptDriftsFinder
    concepts: List[ConceptDriftResult] = ConceptDriftsFinder().find_concept_drifts(transactions, "YearBuilt",
                                                                                   "SalePrice", min_confidence=0.4,
                                                                                   min_support=0.4, diff_threshold=0.1)

    # Convert to dataframe
    concepts_df = pd.DataFrame([x.to_dict() for x in concepts])

    # Print dataframe
    pd.set_option("display.max_columns", 20)
    print(concepts_df.head())
