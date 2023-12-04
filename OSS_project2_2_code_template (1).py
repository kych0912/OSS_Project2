from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

import numpy as np


import pandas as pd

def sort_dataset(dataset_df):
	df_sort = dataset_df.sort_values(by="p_year",ascending=False)
	return df_sort

def split_dataset(dataset_df):	
	df_rescale = dataset_df
	df_rescale.salary = df_rescale.salary.multiply(0.001)

	X = df_rescale.drop(['salary'], axis=1)
	Y = df_rescale['salary']

	X_train = X[:1718]
	X_test = X[1718:]
	Y_train = Y[:1718]
	Y_test = Y[1718:]

	return X_train, X_test, Y_train, Y_test

def extract_numerical_cols(dataset_df):
	return dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war' ]]

def train_predict_decision_tree(X_train, Y_train, X_test):
	dt_cls = DecisionTreeRegressor();
	Y_train_type = Y_train.astype({'salary': 'int'})
	dt_cls.fit(X_train, Y_train)
	dt_predictions = dt_cls.predict(X_test)
	return dt_predictions

def train_predict_random_forest(X_train, Y_train, X_test):
	rf_cls = RandomForestRegressor	();
	Y_train_type = Y_train.astype({'salary': 'int'})
	rf_cls.fit(X_train, Y_train)
	rf_predictions = rf_cls.predict(X_test)
	return rf_predictions


def train_predict_svm(X_train, Y_train, X_test):
	svm_pipe = make_pipeline(
		StandardScaler(),
		SVC()
	)
	Y_train_type = Y_train.astype({'salary': 'int'})
	svm_pipe.fit(X_train, Y_train)
	svm_predictions = svm_pipe.predict(X_test)
	return svm_predictions

def calculate_RMSE(labels, predictions):
	mse = mean_squared_error(labels, predictions)
	rmse = np.sqrt(mse)
	return rmse

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))