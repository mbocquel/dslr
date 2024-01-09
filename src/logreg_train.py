import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sys import argv

def normalize_value(df):
	dfNum = df.select_dtypes(include=['int64','float64'])
	stats = pd.DataFrame(columns=dfNum.columns)
	col_name=stats.columns
	for i in range (len(stats.columns)):
		col = df.loc[:, col_name[i]]
		col_allNum = df.loc[~col.isna(), col_name[i]]
		count = len(col_allNum)
		mean = sum(col_allNum) / len(col_allNum)
		var =  sum([(x - mean)**2 for x in col_allNum])/count
		std = var**(0.5)
		stats.loc['Count', col_name[i]] = count
		stats.loc['Mean', col_name[i]] = mean
		stats.loc['Std', col_name[i]] = std
	dfN = df.copy()
	for i in range(len(df.columns)-1):
		mean = stats.loc["Mean", df.columns[i]]
		std = stats.loc["Std", df.columns[i]]
		dfN.loc[:, df.columns[i]] = (dfN.loc[:, df.columns[i]] - mean) / std
	return dfN, mean, std

def computeCost(X, y, W, b, lam = 1):
	"""
	X (ndarray (m,n): Data, m examples with n features
	y (ndarray (m,)): target values
	w (ndarray (n,)): model parameters
	b (scalar)      : model parameter
	lambda_ (scalar): Controls amount of regularization
	Returns:
		total_cost (scalar):  cost
	"""
	m, n = X.shape
	cost = 0
	for i in range(m):
		z = np.dot(W, X.iloc[i].values) + b
		f_wb = 1 / (1 + np.exp(-z))
		cost += -y.iloc[i].item() * np.log(f_wb) - (1 - y.iloc[i].item())*np.log(1-f_wb)
	cost = cost / m
	reg_part = 0
	for i in range(n):
		reg_part += W[i]**2
	cost = cost + (lam / (2*m)) * reg_part
	return cost

def executeGradientDescentAlgo(X, y, alpha, lambda_, nb_iter):
	m, n = X.shape
	w = np.zeros(n)
	b = 0
	result = []
	for i in range(nb_iter):
		w, b = updateWb(X, y, w, b, lambda_, alpha)
		result.append(computeCost(X, y, w, b, lambda_))
	# x = range(len(result))
	# plt.plot(x, result, 'b')
	# plt.xlabel("Algo iterations")
	# plt.ylabel("Cost")
	# plt.title("Evolution of the Cost with gradient descent iterations")
	# plt.show()
	return(w, b)

def updateWb(X, y, w, b, lambda_, alpha):
	m, n = X.shape
	dj_dw = np.zeros((n,))
	dj_db = 0.0

	for i in range(m):
		z = np.dot(X.iloc[i].values, w) + b
		f_wb_i = 1 / (1 + np.exp(-z))
		err_i  = f_wb_i  - y.iloc[i].item()
		for j in range(n):
			dj_dw[j] = dj_dw[j] + err_i * X.iloc[i,j].item()
		dj_db = dj_db + err_i
	dj_dw = dj_dw/m
	dj_db = dj_db/m

	for j in range(n):
		dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

	W_updated = w - alpha * dj_dw
	b_updated = b - alpha * dj_db
	return (W_updated, b_updated)

def logreg(df, school_name):
	df.loc[:,"Hogwarts House"] == school_name
	df.loc[df.loc[:,"Hogwarts House"] == school_name, :]
	df.loc[df.loc[:,"Hogwarts House"] == school_name, "Test" + school_name] = 1
	df.loc[~(df.loc[:,"Hogwarts House"] == school_name), "Test" + school_name] = 0
	df.drop("Hogwarts House", axis=1, inplace=True)
	df.drop("Astronomy", axis=1, inplace=True)
	dfN, mean, std = normalize_value(df)
	for i in range(len(dfN.columns)):
		dfN = dfN.loc[~dfN.isna()[dfN.columns[i]], :]
	W = np.zeros(len(dfN.columns) - 1)
	X = dfN.iloc[:, :len(dfN.columns)-1]
	y = dfN.iloc[:, len(dfN.columns)-1:]
	alpha = 0.1
	lambda_ = 1
	return (executeGradientDescentAlgo(X, y, 0.3, lambda_, 150))

def main():
	try:
		df = pd.read_csv("../datasets/dataset_train.csv", index_col = "Index")
		assert df is not None, "There is a problem with the dataset..."
		df_bis = df.drop(['First Name', 'Last Name', "Birthday", "Best Hand"], axis=1, inplace=False)
		df = df_bis.copy()
		w_sly, b_sly = logreg(df_bis.copy(), "Slytherin")
		w_rav, b_rav = logreg(df_bis.copy(), "Ravenclaw")
		w_gryf, b_gryf = logreg(df_bis.copy(), "Gryffindor")
		w_huf, b_huf = logreg(df_bis.copy(), "Hufflepuff")
		print("sly stat = ", w_sly, b_sly, "\nrav_stat =", w_rav, b_rav, "\ngryf stat = ", w_gryf, b_gryf, "\nHuf stat = ", w_huf, w_huf)
	except AssertionError as msg:
		print("AssertionError:", msg)
		return 1
	except Exception as err:
		print("Error: ", err)
		return 1

if __name__ == "__main__":
	main()
