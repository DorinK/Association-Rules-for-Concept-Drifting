from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional
import pandas as pd
import numpy as np


@dataclass
class TrainParams:
    na_columns_mean: Dict[int, float]
    numerical_columns_cut: Dict[str, List[float]]
    dropped_columns: List[str]
    numeric_columns: List[str]
    very_numerical: List[str]
    categorical_columns: List[str]
    too_many_categories_columns: List[str]
    ordinals: List[str]


def preprocess_dataset(df, train_params: Optional[TrainParams] = None):
    """
    Helper function to preprocess datasets (based on notebook from class)
    """

    # Defining numeric and categorical columns
    if train_params is None:
        numeric_columns = df.dtypes[(df.dtypes == "float64") | (df.dtypes == "int64")].index.tolist()
        very_numerical = [nc for nc in numeric_columns if df[nc].nunique() > 20]
        categorical_columns = [c for c in df.columns if c not in numeric_columns]
        too_many_categories_columns = [c for c in categorical_columns if df[c].nunique() > 20]
        categorical_columns = list(set(categorical_columns) - set(too_many_categories_columns))
        ordinals = list(set(numeric_columns) - set(very_numerical))
    else:
        numeric_columns = train_params.numeric_columns
        very_numerical = train_params.very_numerical
        categorical_columns = train_params.categorical_columns
        too_many_categories_columns = train_params.too_many_categories_columns
        ordinals = train_params.ordinals
        dropped_columns = train_params.dropped_columns

    # Filling Null Values with the column's mean
    na_columns = df[very_numerical].isna().sum()
    na_columns = na_columns[na_columns > 0]
    na_columns_mean = {}

    for nc in na_columns.index:
        if train_params is None:
            column_mean = df[nc].mean()

            # Save mean
            na_columns_mean[nc] = column_mean
        else:
            column_mean = train_params.na_columns_mean[nc]

        df[nc].fillna(column_mean, inplace=True)

    # Dropping and filling NA values for categorical columns:
    if train_params is None:
        # drop if at least 70% are NA:
        nul_cols = df[categorical_columns].isna().sum() / len(df)
        drop_us = nul_cols[nul_cols > 0.7]
        df = df.drop(drop_us.index, axis=1)
        dropped_columns = [d for d in drop_us.keys()]
    else:
        df = df.drop(columns=dropped_columns)

    # Drop columns that have too many categories
    if train_params is None:
        df = df.drop(columns=list(set(too_many_categories_columns) - set(dropped_columns)))
        dropped_columns += too_many_categories_columns
        dropped_columns = list(set(dropped_columns))

    # Fill with a new 'na' category:
    if train_params is None:
        categorical_columns = list(set(categorical_columns) - set(dropped_columns))
    # df[categorical_columns] = df[categorical_columns].fillna('na')

    # Fill Null values in ordinals with a new '-1' ordinal:
    df[ordinals] = df[ordinals].fillna(-1)

    df = df.copy()

    # Bin numerical data
    numerical_columns_cut = {}
    for c in very_numerical:
        if train_params is None:
            try:
                # df[c] = pd.qcut(df[c], 5, labels=["very low", "low", "medium", "high", "very high"])
                df[c], bins = pd.qcut(df[c], 5, labels=[1, 2, 3, 4, 5], retbins=True)
            except:
                # sometimes for highly skewed data, we cannot perform qcut as most quantiles are equal
                # df[c] = pd.cut(df[c], 5, labels=["very low", "low", "medium", "high", "very high"])
                df[c], bins = pd.cut(df[c], 5, labels=[1, 2, 3, 4, 5], retbins=True)

            # Make bin edges larger (infinity and -infinity)
            bins = np.concatenate(([-np.inf], bins[1:-1], [np.inf]))

            # Save bin
            numerical_columns_cut[c] = bins

        else:
            # Use existing train bins
            bins = train_params.numerical_columns_cut[c]
            df[c] = pd.cut(df[c], labels=[1, 2, 3, 4, 5], bins=bins)

    return df, TrainParams(na_columns_mean, numerical_columns_cut, dropped_columns,
                           numeric_columns, very_numerical, categorical_columns,
                           too_many_categories_columns, ordinals)


def split_X_y(df_prep: pd.DataFrame, columns_to_use: List[str], train_params: TrainParams, one_hot_columns: List[str],
              target_column) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_columns = list(set(columns_to_use) - set(train_params.dropped_columns) - {target_column})
    X = df_prep[X_columns]

    for one_hot_column in list(set(one_hot_columns) - set(train_params.dropped_columns)):
        X = pd.concat([X, pd.get_dummies(X[one_hot_column], prefix=one_hot_column)], axis=1)
        X = X.drop(columns=[one_hot_column])

    y = df_prep[target_column]

    return X, y
