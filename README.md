# Association-Rules-for-Concept-Drifting
Final project in 'Tabular Data Science' course by Dr. Amit Somech at Bar-Ilan University.

## ConceptDriftsFinder

In the Exploratory Data Analysis process (EDA), `ConceptsDriftsFinder` can be used to automatically find concept drifts.
The `find_concept_drifts` function receives  a list of transactions and returns a list of `ConceptDriftResult` objects.

### Example usage
Let's say we suspect the column `OverallQual` as a concept drift, we can run:

```python
ConceptsDriftFinder().find_concept_drifts(transactions, concept_column="OverallQual", target_column="SalePrice")
```

Here is a sample of the output in a table format:

|    | left_hand_side                      | right_hand_side   |   confidence_before |   confidence_after |   support_before |   support_after |   lift_before |   lift_after |   concept_cutoff | concept_column   |
|---:|:------------------------------------|:------------------|--------------------:|-------------------:|-----------------:|----------------:|--------------:|-------------:|-----------------:|:-----------------|
|  0 | {'BldgType': '1Fam'}                | {'SalePrice': 1}  |                   1 |           0.185229 |              1   |        0.155359 |             1 |     0.941887 |              2.8 | OverallQual      |
|  1 | {'FullBath': 1}                     | {'SalePrice': 1}  |                   1 |           0.374718 |              0.6 |        0.163225 |             1 |     1.90544  |              2.8 | OverallQual      |
|  2 | {'GrLivArea': 1}                    | {'SalePrice': 1}  |                   1 |           0.53     |              1   |        0.104228 |             1 |     2.69505  |              2.8 | OverallQual      |
|  3 | {'YearBuilt': 1}                    | {'SalePrice': 1}  |                   1 |           0.509615 |              0.8 |        0.104228 |             1 |     2.59139  |              2.8 | OverallQual      |

We can see that if `OverallQual<2.8` then `BldgType=1Fam` becomes a more important indication of `SalePrice`.
This is useful to better understand our dataset.


### Notes

See the following section about preprocessing your data before you can use `ConceptDriftsFinder`.

The amount of drifts found can be controlled with the following parameters: `min_confidence: float`,
`min_support: float`, `diff_threshold: float`.

A `pandas.DataFrame` object can be converted to transactions using the helper function `convert_df_to_transactions`.

### Preprocessing

When working with association rules, we can't use numerical values, only categorical or ordinal.
`preprocessing.py` contains code to convert numerical
values to ordinals, as well as data cleaning such as dropping N/A.


### CutoffValuesFinder

This class is an inner class and shouldn't be used directly.

The `CutoffValuesFinder` classifies each concept as discrete or continuous, and based on that it decides for
`ConceptDriftsFinder` which concept values to try.

## ConceptEngineering
To easily start using this library for feature engineering for machine learning models, we created `ConceptEngineering`.
The idea is to automatically take the found concept drifts into account by changing the values of the dataset.

### Example usage
Let's look at the following row from our dataset (shortened for readability):

| Id              |   1101 |
|:----------------|-------:|
| YearBuilt       |      1 |
| FullBath        |      1 |
| OverallQual     |      2 |
| BldgType_1Fam   |      1 |
| BldgType_2fmCon |      0 |
| BldgType_Duplex |      0 |

If we continue with our example above, we know that if `OverallQual<2.8` then `BldgType=1Fam` becomes a more
important indication for `SalePrice=1`.  
We can help the model use this information by increasing the weight of `BldgType=1Fam` whenever `OverallQual<2.8`, which
can be especially helpful when using models such as LogisticRegression which have a single weight per feature.

We can run:
```python
# Find association rules
df_prep, train_params = preprocess_dataset(df)
X, y = split_X_y(df_prep, columns, train_params, one_hot_columns, target_column)
concept_engineering = ConceptEngineering()
X = concept_engineering.fit_transform(X, df, target_column, one_hot_columns)
```

Now our new `X` dataframe will have increased weights based on all of the found rules, for example:

| Id              |    1101 |
|:----------------|--------:|
| YearBuilt       | 1.37681 |
| FullBath        | 1.2922  |
| OverallQual     | 2       |
| BldgType_1Fam   | 0.94262 |
| BldgType_2fmCon | 0       |
| BldgType_Duplex |       0 |

To calculate the change to the value, we use the difference between the lift values.

## Example notebooks
We run 4 different datasets with our Concept Drifting finder using association rules. See the `notebooks/` directory.


### Datasets

To test our library, we used 4 datasets:
* [Housing prices](https://github.com/amitsomech/TDS-COURSE/tree/master/datasets/houseprices).
* [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
* [Big Mart Sales](https://www.kaggle.com/akashdeepkuila/big-mart-sales?select=Train-Set.csv)
* [Latest Netflix data with 26+ joined attributes](https://www.kaggle.com/ashishgup/netflix-rotten-tomatoes-metacritic-imdb)

