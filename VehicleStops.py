
import os
import io
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import *

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def create_cat_feats():
    """
    :Example:
    >>> pl = create_cat_feats()
    >>> isinstance(pl, Pipeline)
    True
    >>> fp = os.path.join('data', 'sample_stops.csv')
    >>> stops = pd.read_csv(fp)[['subject_sex']]
    >>> pl.fit(stops) # doctest:+ELLIPSIS
    Pipeline(...)
    >>> pl.transform(stops.iloc[0:1]).toarray().tolist()[0] == [0, 1, 0]
    True
    """

    return Pipeline([
    		('string-ify', FunctionTransformer(func = lambda x : x.astype(str), validate = False)),
    		('simple-impute', SimpleImputer(strategy = 'constant', fill_value = 'NULL')),
    		('one-hot', OneHotEncoder(handle_unknown = 'ignore'))
	])

# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def round_to_five(nums):
	to_return = []
	for num in nums:
		if num % 5 == 0:
			to_return.append(num)
		elif num % 5 > 2:
			to_return.append(num + (5 - num % 5))
		else:
			to_return.append(num - num % 5)
	return np.array(to_return)

def create_age_feats():
    """
    :Example:
    >>> pl = create_age_feats()
    >>> isinstance(pl, Pipeline)
    True
    >>> fp = os.path.join('data', 'sample_stops.csv')
    >>> stops = pd.read_csv(fp)[['subject_age']]
    >>> pl.fit(stops) # doctest:+ELLIPSIS
    Pipeline(...)
    >>> pl.transform(stops.iloc[0:1])[0][0] == 20
    True
    """

    return Pipeline([
    	('fillna', SimpleImputer(strategy = 'mean')),
    	('move-to-five', FunctionTransformer(func = round_to_five, validate = False))
	])



# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def dayofweek_and_hour(df):
    return df[['day of week', 'hour']]

def baseline_model():
    """
    :Example:
    >>> pl = baseline_model()
    >>> isinstance(pl, Pipeline)
    True
    >>> isinstance(pl.steps[-1][-1], LogisticRegression)
    True
    >>> fp = os.path.join('data', 'sample_stops.csv')
    >>> stops = pd.read_csv(fp).drop('searched', axis=1)
    >>> searched_fp = os.path.join('data', 'sample_imp_search.csv')
    >>> searched = pd.read_csv(searched_fp, names=['searched'], squeeze=True)
    >>> pl.fit(stops.iloc[:500], searched.iloc[:500]) # doctest:+ELLIPSIS
    Pipeline(...)
    >>> out = pl.predict(stops)
    >>> pd.Series(out).isin([0,1]).all()
    True
    """
    cat_feat = ['stop_cause', 'service_area', 'subject_race', 'subject_sex', 'sd_resident']
    smoothed_feat = ['subject_age']
    unchanged_feat = ['dayofweek', 'hour']
    pl_cat_feat = create_cat_feats()
    pl_smoothed_feat = create_age_feats()
    preproc = ColumnTransformer(transformers=
                            [('cat', pl_cat_feat, cat_feat), 
                             ('smoothed', pl_smoothed_feat, smoothed_feat),
                             ('unchanged', 'passthrough', unchanged_feat)])
    return Pipeline(steps=[('preprocessor', preproc), ('regressor', LogisticRegression())])


def train_test_acc(pl, stops, searched):
    """
    :Example:
    >>> pl = baseline_model()
    >>> fp = os.path.join('data', 'sample_stops.csv')
    >>> stops = pd.read_csv(fp)
    >>> searched_fp = os.path.join('data', 'sample_imp_search.csv')
    >>> searched = pd.read_csv(searched_fp, names=['searched'], squeeze=True)
    >>> out = train_test_acc(pl, stops, searched)
    >>> np.isclose(out, 0.90, 0.1).all()
    True
    """
    df = stops.copy().drop('stop_id', axis = 1)
    if 'searched' in df.columns:
        X = df.drop('searched', axis = 1)
    else:
        X = df
    y = searched
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    pl.fit(X_train, y_train)
    return (pl.score(X_train, y_train), pl.score(X_test, y_test))


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def constant_model_acc(searched):
    """
    :Example:
    >>> searched_fp = os.path.join('data', 'sample_imp_search.csv')
    >>> searched = pd.read_csv(searched_fp, names=['searched'], squeeze=True)
    >>> 0.8 <= constant_model_acc(searched) <= 1.0
    True
    """
    predict = [0] * len(searched)
    return accuracy_score(predict, searched)


def model_outcomes(predictions, target):
    """
    :Example:
    >>> out = model_outcomes(pd.Series([1,0,1,0]), pd.Series([0,1,1,0]))
    >>> (np.diag(out) == 1).all()
    True
    >>> set(out.columns) == {'FN', 'FP', 'TN', 'TP'}
    True
    """
    df = pd.DataFrame()
    df['FP'] = [1 if predictions.loc[x] == 1 and target.loc[x] == 0 else 0 for x in range(len(predictions))]
    df['FN'] = [1 if predictions.loc[x] == 0 and target.loc[x] == 1 else 0 for x in range(len(predictions))]
    df['TP'] = [1 if predictions.loc[x] == 1 and target.loc[x] == 1 else 0 for x in range(len(predictions))]
    df['TN'] = [1 if predictions.loc[x] == 0 and target.loc[x] == 0 else 0 for x in range(len(predictions))]
    return df


def metrics(predictions, target):
    """
    :Example:
    >>> out = metrics(pd.Series([1,0,1,0]), pd.Series([0,1,1,0]))
    >>> set(out.index) == {'acc', 'f1', 'fdr', 'fnr', 'fpr', 'precision', 'recall', 'specificity'}
    True
    >>> (out == 0.5).all()
    True
    """
    index = ['acc', 'f1', 'fdr', 'fnr', 'fpr', 'precision', 'recall', 'specificity']
    tn, fp, fn, tp = (confusion_matrix(predictions, target, labels=[0,1]).ravel())
    values = [accuracy_score(predictions, target), 
            recall_score(predictions, target),
            recall_score(predictions, target, pos_label=0),
            precision_score(predictions, target),
            fn / (fn + tn),
            fp / (fp + tp),
            fp / (tp + fp),
            f1_score(predictions, target)
            ]

    return pd.Series(values, index = index)

# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------


class AdditiveSmoother(BaseEstimator, ClassifierMixin):
    """
    :Example:

    >>> fp = os.path.join('data', 'sample_stops.csv')
    >>> stops = pd.read_csv(fp)
    >>> searched_fp = os.path.join('data', 'sample_imp_search.csv')
    >>> searched = pd.read_csv(searched_fp, names=['searched'], squeeze=True)
    >>> asm = AdditiveSmoother()
    >>> asm.fit(stops[['subject_sex']], searched)
    AdditiveSmoother(alpha=100)
    >>> np.isclose(asm.srate, 0.054)
    True
    >>> internal = asm.smdists['subject_sex']['M']
    >>> out = asm.transform(stops[['subject_sex']].iloc[[0]])[0][0]
    >>> np.isclose(internal, out)
    True
    """

    def __init__(self, alpha=100):
        self.alpha = alpha

    def fit(self, X, y, **kwargs):
        """
        Calculates the smoothed condition empirical 
        distributions of the columns of X dependent on y.
        In this case, y is searches in the stops data.
        """

        # calculate the search rate
        X = pd.DataFrame(X)
        self.srate = np.mean(y)

        smdists = {}
        df = X.copy()
        df['searched'] = y
        # loop through the columns of X
        for c in X.columns:
            # create a smoothed empirical distribution for each column in X
            smoothed = df.groupby(c)['searched'].mean().to_dict()
            toSeries = pd.Series(smoothed)
            values = X[c].value_counts()
            smoothedSeries = (toSeries * values + self.alpha * self.srate) / (values + self.alpha)
            smoothedDict = smoothedSeries.to_dict()
            smdists[c] = smoothed
        # smoothed empirical search rates in smdists
        self.smdists = smdists
        return self

    def transform(self, X):
        """
        Transforms the categorical values in the columns of X to
        the smoothed search rates of those values.
        """
        X = pd.DataFrame(X)
        toReturn = []
        for i in range(len(X)):
            smoothed = []
            row = X.loc[i]
            for c in X.columns:
                if row[c] in self.smdists[c]:
                    val = self.smdists[c][row[c]]
                else:
                    val = self.srate
                smoothed.append(val)
            toReturn.append(smoothed)
        return np.array(toReturn)

    def get_params(self, deep=False):
        """
        Gets the parameters of the transformer;
        Allows Gridsearch to be used with class.
        """
        return {'alpha': self.alpha}


# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------


def final_model():
    """
    :Example:
    >>> pl = final_model()
    >>> isinstance(pl, Pipeline)
    True
    >>> isinstance(pl.steps[-1][-1], BaseEstimator)
    True
    >>> fp = os.path.join('data', 'sample_stops.csv')
    >>> stops = pd.read_csv(fp).drop('searched', axis=1)
    >>> searched_fp = os.path.join('data', 'sample_imp_search.csv')
    >>> searched = pd.read_csv(searched_fp, names=['searched'], squeeze=True)
    >>> pl.fit(stops.iloc[:500], searched.iloc[:500]) # doctest:+ELLIPSIS
    Pipeline(...)
    >>> out = pl.predict(stops)
    >>> pd.Series(out).isin([0,1]).all()
    True

    """
    cat_feat = ['stop_cause', 'service_area', 'subject_race', 'subject_sex', 'sd_resident']
    smoothed_feat = ['subject_age']
    unchanged_feat = ['dayofweek', 'hour']
    pl_cat_feat = Pipeline([
            ('string-ify', FunctionTransformer(func = lambda x : x.astype(str), validate = False)),
            ('simple-impute', SimpleImputer(strategy = 'constant', fill_value = 'NULL')),
            ('add-smooth', AdditiveSmoother())
    ])
    pl_smoothed_feat = create_age_feats()
    preproc = ColumnTransformer(transformers=
                            [('cat', pl_cat_feat, cat_feat),
                             ('smoothed', pl_smoothed_feat, smoothed_feat),
                             ('unchanged', 'passthrough', unchanged_feat)])
    return Pipeline(steps=[('preprocessor', preproc), ('tree', RandomForestClassifier())])

# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def compare_search_rate(stops, predictions, col):
    """
    :Example:
    >>> fp = os.path.join('data', 'sample_stops.csv')
    >>> stops = pd.read_csv(fp)
    >>> randpred = np.random.choice([0,1], size=len(stops))
    >>> out = compare_search_rate(stops, randpred, 'hour')
    >>> set(out.columns) == {'searched', 'predicted'}
    True
    >>> (out.index == range(24)).all()
    True
    """
    df = stops.copy()
    df['predicted'] = predictions
    df = df.dropna(subset = ['searched'])
    df = df.groupby(col)[['searched', 'predicted']].mean()
    return df


def compare_metrics(stops, predictions, col):
    """
    :Example:
    >>> fp = os.path.join('data', 'sample_stops.csv')
    >>> stops = pd.read_csv(fp)
    >>> randpred = np.random.choice([0,1], size=len(stops))
    >>> out = compare_metrics(stops, randpred, 'hour')
    >>> 'precision' in out.columns
    True
    >>> (out.index == range(24)).all()
    True

    """
    df = stops.copy()
    df['predicted'] = predictions
    df = df.dropna(subset = ['searched'])
    df = df.groupby(col).apply(lambda x : metrics(x['predicted'], x['searched']))
    return df



# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['create_cat_feats'],
    'q02': ['create_age_feats'],
    'q03': ['baseline_model','train_test_acc'],
    'q04': ['constant_model_acc', 'model_outcomes', 'metrics'],
    'q05': ['AdditiveSmoother'],
    'q06': ['final_model'],
    'q07': ['compare_search_rate', 'compare_metrics']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
