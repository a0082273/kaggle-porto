# Library Importing
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression



# Data Loading
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

id_test = test['id'].values
target_train = train['target'].values

train = train.drop(['target','id'], axis = 1)
test = test.drop(['id'], axis = 1)



# Feature Engineering
# Feature creating


# Feature selection
col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)
test = test.drop(col_to_drop, axis=1)


# NaN filling
train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)


# Categorical to dummy
cat_features = [a for a in train.columns if a.endswith('cat')]

for column in cat_features:
        temp = pd.get_dummies(pd.Series(train[column]))
        train = pd.concat([train,temp],axis=1)
        train = train.drop([column],axis=1)

for column in cat_features:
        temp = pd.get_dummies(pd.Series(test[column]))
        test = pd.concat([test,temp],axis=1)
        test = test.drop([column],axis=1)

print(train.values.shape, test.values.shape)



# Modeling
class Stack(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        score_cv = np.zeros((len(self.base_models), self.n_splits))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]

                print ("Fit model_%d fold_%d" % (i+1, j+1))
                clf.fit(X_train, y_train)
                y_pred = clf.predict_proba(X_holdout)[:,1]
                score_cv[i, j] = roc_auc_score(y_holdout, y_pred)
                print("Cross validation score: %.5f" % (score_cv[i, j]))

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)
            print("Average of cross validation score: %.5f" % (score_cv[i, :].mean()))

        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res


# Parameter tuning


# Model creating
lgb1_params = {}
lgb1_params['learning_rate'] = 0.02
lgb1_params['n_estimators'] = 650
lgb1_params['max_bin'] = 10
lgb1_params['subsample'] = 0.8
lgb1_params['subsample_freq'] = 10
lgb1_params['colsample_bytree'] = 0.8
lgb1_params['min_child_samples'] = 500
lgb1_params['seed'] = 99

lgb2_params = {}
lgb2_params['n_estimators'] = 1090
lgb2_params['learning_rate'] = 0.02
lgb2_params['colsample_bytree'] = 0.3
lgb2_params['subsample'] = 0.7
lgb2_params['subsample_freq'] = 2
lgb2_params['num_leaves'] = 16
lgb2_params['seed'] = 99

lgb3_params = {}
lgb3_params['n_estimators'] = 1100
lgb3_params['max_depth'] = 4
lgb3_params['learning_rate'] = 0.02
lgb3_params['seed'] = 99

lgb1 = LGBMClassifier(**lgb1_params)
lgb2 = LGBMClassifier(**lgb2_params)
lgb3 = LGBMClassifier(**lgb3_params)
log = LogisticRegression()


# Ensembling
stack = Stack(n_splits=3,
        stacker = log,
        base_models = (lgb1, lgb2, lgb3))


# Predicting
y_pred = stack.fit_predict(train, target_train, test)


# Submission file making
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('submission.csv', index=False)
