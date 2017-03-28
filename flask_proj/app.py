from flask import Flask, request, session, g, redirect, url_for, abort, \
	 render_template, flash

import re
import numpy as np
import pandas as pd
from nvd3 import lineChart
import mpld3
# import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

# from sklearn.model_selection import cross_val_predict
# from sklearn import linear_model


app = Flask(__name__)
app.secret_key = 'test123'
app.config['DEBUG'] = True



def parse_str_to_df(input_str):	
	#each row is now an element in input_list
	input_list = re.split("\n", input_str) 

	header = input_list[0].split("\t")
	
	table = []
	for i in input_list:
		#preventing carriage returns
		i = i.rstrip()
		table.append(i.split("\t"))
	table = table[1:]

	input_df = pd.DataFrame(table, columns=header)
	return input_df

def rf_pred(X_train, X_test, y_train, y_test):
	clf = RandomForestClassifier(n_estimators=100)
	rf_clf = clf.fit(X_train, y_train)
	y_pred = rf_clf.predict(X_test)
	results_df = pd.DataFrame()
	results_df['y_test'] = y_test
	results_df['y_pred_random_forest'] = y_pred
	#results_df['actual - predicted'] = results_df['y_test'] - results_df['y_pred_random_forest']
	return y_pred
	#return results_df

def lin_pred(X_train, X_test, y_train, y_test):
	# clf = RandomForestClassifier(n_estimators=100)
	# rf_clf = clf.fit(X_train, y_train)
	# y_pred = rf_clf.predict(X_test)
	from sklearn.linear_model import LinearRegression

	lin_reg = LinearRegression()
	lin_reg_est = lin_reg.fit(X_train, y_train)
	y_pred = lin_reg_est.predict(X_test)
	return y_pred

	# results_df = pd.DataFrame()
	# results_df['y_test'] = y_test
	# results_df['y_pred_random_forest'] = y_pred
	# #results_df['actual - predicted'] = results_df['y_test'] - results_df['y_pred_random_forest']
	# return y_pred




def make_scatter_plot(X_test, y_test, y_pred_rf, y_pred_lin):
	fig, ax = plt.subplots()
	ax.scatter(list(X_test), list(y_test), color='b', label='true')
	ax.scatter(list(X_test), list(y_pred_rf), color='r', label='rf')
	ax.scatter(list(X_test), list(y_pred_lin), color='g', label='lin')
	ax.grid(color='lightgray', alpha=0.7)
	chart = mpld3.display(fig)
	return chart


#------------------------------------------------------------------------VIEWS-From--the--6------------------------------------------------------------


@app.route("/", methods=['GET', 'POST'])
def index():
	return render_template('user_input.html')



@app.route("/submit", methods=['GET', 'POST'])
def submit():
	input_str = request.form.get('text_area_field')

	input_df = parse_str_to_df(input_str)
	X = input_df[input_df.columns[0]]
	X = X[:, None]
	y = input_df[input_df.columns[-1]]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	html_table_input = input_df.to_html()

	#results_df = rf_pred(X_train, X_test, y_train, y_test)
	y_pred_rf = rf_pred(X_train, X_test, y_train, y_test)

	y_pred_lin = lin_pred(X_train, X_test, y_train, y_test)

	results_df = pd.DataFrame()
	results_df['y_test'] = y_test
	results_df['y_pred_random_forest'] = y_pred_rf
	results_df['y_pred_lin_reg'] = y_pred_lin


	html_table_results = results_df.to_html()

	chart = make_scatter_plot(X_test, y_test, y_pred_rf, y_pred_lin)

	return render_template('success.html', html_table_input=html_table_input, html_table_results=html_table_results,chart=chart)





if __name__ == "__main__":
	app.run()





