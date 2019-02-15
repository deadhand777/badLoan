## import libraries

#load packages
import os as os

import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__))

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__))

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)

## import data modelling libs
#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from pprint import pprint

np.random.seed(0)

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

# Display up to 60 columns of a dataframe
pd.set_option('display.max_columns', 60)

mpl.style.use('ggplot')
# Set default font size
plt.rcParams['font.size'] = 24

sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


# 2 import data
## read data from csv
data = pd.read_csv(os.getcwd() + '/data//Data_Import_Template_v1.0-1.csv', sep=";")

# create copy
data_copy = data.copy(deep=True)

# preview data
data.info()
data.head()
data.sample()

# Tidy data
data.describe(include = 'all')

# Correct data types
for col in data.columns.tolist():
    if data[col].dtype =='object':
        data[col] = pd.Categorical(data[col])
    else:
        data[col] = data[col].astype(float)

#data_copy[['JOB', 'REASON']] = data_copy[['JOB', 'REASON']].apply(lambda x: pd.Categorical(x))

# set y and X
y = data['BAD']
X = data.drop(columns=['BAD'], axis=1)

# replace missing values in categorical variables

def replace_missing_values(df = data):
  for col in data.columns.tolist():
    if ('REASON' in col or 'JOB' in col):
      data[col] = data[col].cat.add_categories('MISSING').fillna('MISSING')
  return(data)    

X = replace_missing_values(df = X)  

#data_copy.assign(JOB = data_copy.JOB.cat.add_categories('Missing').fillna('Missing'))

# train test split

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                    y,
                                                    test_size = 0.25,
                                                    #stratify = y,
                                                    random_state=0)
                                                    
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_features = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['JOB', 'REASON']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


pipe_logistic_regression = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('logReg', LogisticRegression())])

pipe_logistic_regression.fit(X_train, y_train)
print("model score: %.3f" % pipe_logistic_regression.score(X_test, y_test))

# xgboost pipeline
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBClassifier
pipe_XGBoost = Pipeline(steps=[('preprocessor', preprocessor),
                               ('xgboost', XGBClassifier())])

pipe_XGBoost.fit(X_train, y_train)
print("model score: %.3f" % pipe_XGBoost.score(X_test, y_test))

# neural net
from sklearn.neural_network import MLPClassifier
pipe_NN = Pipeline(steps=[('preprocessor', preprocessor),
                               ('nn', MLPClassifier())])
pipe_NN.fit(X_train, y_train)                               
print("model score: %.3f" % pipe_NN.score(X_test, y_test))

# ROC

y_predicted_lg = pipe_logistic_regression.predict_proba(X_test)[:,1].tolist()
y_predicted_xg = pipe_XGBoost.predict_proba(X_test)[:,1].tolist()
y_predicted_nn = pipe_NN.predict_proba(X_test)[:,1].tolist()

model_list = [
    {
        'legend_names' : 'LogReg',
        'y_empirical': y_test,
        'y_predicted': y_predicted_lg,
    },
    {
        'legend_names' : 'xgBoost',
        'y_empirical': y_test,
        'y_predicted': y_predicted_xg,        
    },
    {
        'legend_names' : 'NN',
        'y_empirical': y_test,
        'y_predicted': y_predicted_nn,
    }        
]

def compare_model(models = model_list):
    for m in models:
        false_positive_rate, true_positive_rate, thresholds = roc_curve(m['y_empirical'], m['y_predicted'])
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, label='%s ROC (area = %0.2f)' % (m['legend_names'], roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", fontsize='small')
    plt.show()
    
# Hyperparameter optimization

# check hyperparameters
pprint(pipe_logistic_regression.fit(X_train, y_train).get_params())    

params={'logReg__C':[.01,.05,.1,.5,1,5,10],
        'logReg__penalty':['l1','l2']}
        
# Create hyperparameter options

# use hyperparameter grid from above in Cross Validation

CV_logReg  = GridSearchCV(pipe_logistic_regression, param_grid=params, cv=5, verbose=0)

# fit the model
bestModel_logReg = CV_logReg.fit(X_train, y_train)

print('Best Penalty:', bestModel_logReg.best_estimator_.get_params()['logReg__penalty'])
print('Best C:', bestModel_logReg.best_estimator_.get_params()['logReg__C'])


best_logReg = bestModel_logReg.best_estimator_
best_logReg.fit(X_train,y_train)
best_logReg.coef_ = bestlogreg.named_steps['logReg'].coef_
best_logReg.score(X_train,y_train)
pipe_logistic_regression.score(X_test, y_test)

# XgBoost
pprint(pipe_XGBoost.fit(X_train, y_train).get_params().keys())    

params = {'xgboost__min_child_weight': [1, 5, 10],
        'xgboost__gamma': [0.5, 1, 1.5, 2, 5],
        'xgboost__subsample': [0.6, 0.8, 1.0],
        'xgboost__colsample_bytree': [0.6, 0.8, 1.0],
        'xgboost__max_depth': [3, 4, 5]}

CV_Xgboost  = GridSearchCV(pipe_XGBoost, param_grid=params, cv=5, verbose=0)
bestModel_Xgboost = CV_Xgboost.fit(X_train, y_train)

print('Best Min Child Weight:', bestModel_Xgboost.best_estimator_.get_params()['xgboost__min_child_weight'])
print('Best Gamma:', bestModel_Xgboost.best_estimator_.get_params()['xgboost__gamma'])
print('Best Colsample Bytree:', bestModel_Xgboost.best_estimator_.get_params()['xgboost__colsample_bytree'])
print('Best Max Depth:', bestModel_Xgboost.best_estimator_.get_params()['xgboost__max_depth'])

best_Xgboost = bestModel_Xgboost.best_estimator_
best_Xgboost.fit(X_train,y_train)
best_Xgboost.coef_ = bestlogreg.named_steps['logReg'].coef_
best_Xgboost.score(X_train,y_train)
