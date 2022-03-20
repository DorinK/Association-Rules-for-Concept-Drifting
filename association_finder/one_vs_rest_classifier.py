import pandas as pd
from sklearn.linear_model import LogisticRegression


class OneVsRestClassifier:
    """
    This is a simple scikit-learn LogisticRegression implementation with one-vs-rest (ovr) (i.e., one model for each
    label).
    The only difference in this implementation (instead of using multi_class="ovr") is that we can
    inject different dataset changes for each label classifier (called label_to_transformation).
    This is necessary, because concept rules are per label, and we want to activate the label rules only when
    classifying the relevant label.
    """

    def __init__(self, label_to_transformation: dict = None):
        """
        :param label_to_transformation: For each label, you can specify a function that can transform the dataset
        """

        self.label_to_transformation = label_to_transformation if label_to_transformation is not None else {}
        self.classifiers = {}
        self.train_only_one_model = None

    def fit(self, X, y):
        unique_labels = list(y.unique())

        self.train_only_one_model = len(unique_labels) == 2
        if not self.train_only_one_model:
            for label in unique_labels:
                y_ova = pd.Series(index=y.index, dtype='float64')
                y_ova[y != label] = 0
                y_ova[y == label] = 1

                transformed_X = X
                if label in self.label_to_transformation:
                    transformed_X = self.label_to_transformation[label](X)

                clf = LogisticRegression(random_state=0, max_iter=100000).fit(transformed_X, y_ova)
                self.classifiers[label] = clf
        else:
            transformed_X = X
            for transformation in self.label_to_transformation.values():
                transformed_X = transformation(transformed_X)
            clf = LogisticRegression(random_state=0, max_iter=100000).fit(transformed_X, y)
            self.classifiers[0] = clf

        return self

    def transform(self, X):
        if not self.train_only_one_model:
            y_pred_df = pd.DataFrame(index=X.index, dtype='float64')
            for label in self.classifiers.keys():
                transformed_X = X
                if label in self.label_to_transformation:
                    transformed_X = self.label_to_transformation[label](X)

                y_pred_df[label] = self.classifiers[label].predict_proba(transformed_X)[:, 1]

            y_pred = y_pred_df.apply(lambda row: y_pred_df.columns[row.argmax()], axis=1)
        else:
            transformed_X = X
            for transformation in self.label_to_transformation.values():
                transformed_X = transformation(transformed_X)

            y_pred = self.classifiers[0].predict(transformed_X)

        return y_pred

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


def label_to_concept_transform_wrapper(concept_engineering, target_column, label):
    """
    Wrapper for OneVsRestClassifier with ConceptEngineering transformation
    """

    def label_to_concept_transform(X):
        return concept_engineering.transform(X, filter_concepts_by_target=label, target_column=target_column)

    return label_to_concept_transform
