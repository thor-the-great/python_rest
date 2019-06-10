import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib


dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

data = pd.read_csv(dataset_url, sep=';')


print(data.head())

# print(data.shape)
print(data.describe())


y = data.quality
X = data.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=124,
                                                    stratify=y)

# scaler = preprocessing.StandardScaler().fit(X_train)
#
# X_train_scaled = scaler.transform(X_train)
#
# print(X_train_scaled.mean(axis=0))
#
#
# print(X_train_scaled.std(axis=0))
#
# X_test_scaled = scaler.transform(X_test)
#
# print(X_test_scaled.mean(axis=0))
# # [ 0.02776704  0.02592492 -0.03078587 -0.03137977 -0.00471876 -0.04413827
# #  -0.02414174 -0.00293273 -0.00467444 -0.10894663  0.01043391]
#
# print(X_test_scaled.std(axis=0))
# # [ 1.02160495  1.00135689  0.97456598  0.91099054  0.86716698  0.94193125
# #  1.03673213  1.03145119  0.95734849  0.83829505  1.0286218 ]
#
# pipeline = make_pipeline(preprocessing.StandardScaler(),
#                          RandomForestRegressor(n_estimators=100))
#
# hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
#                   'randomforestregressor__max_depth': [None, 5, 3, 1]}
#
# clf = GridSearchCV(pipeline, hyperparameters, cv=10)
#
# # Fit and tune model
# clf.fit(X_train, y_train)
#
#
# print (clf.best_params_)

print(X_test)

clf = joblib.load('rf_regressor.pkl')

# Predict a new set of dataPython
y_pred = clf.predict(X_test)

print(r2_score(y_test, y_pred))
# 0.45044082571584243

print(mean_squared_error(y_test, y_pred))
# 0.35461593750000003

# 10. Save model for future use
joblib.dump(clf, 'rf_regressor.pkl')
# To load: clf2 = joblib.load('rf_regressor.pkl')