from sys import argv
import pandas as pd
import numpy as np

def calc_proba(valuesN, w, b):
	m, n = valuesN.shape
	proba = np.zeros(m)
	for i in range(m):
		z = np.dot(w, valuesN.iloc[i].values) + b
		proba[i] = 1 / (1 + np.exp(-z))
	return proba


def normalize_value(df_test, df_result):
	test_N = df_test.copy()

	for i in range(len(df_test.columns)):
		mean = df_result.loc["Mean", df_test.columns[i]]
		std = df_result.loc["Std", df_test.columns[i]]
		test_N.loc[:, test_N.columns[i]] = (test_N.loc[:, test_N.columns[i]] - mean) / std
	return (test_N)

def create_prob_arr(df_test, df_result):
	normalized_val = normalize_value(df_test, df_result)
	df_test.loc[:, "Slytherin"] = calc_proba(normalized_val, df_result.loc["Slytherin", :"Flying"], df_result.loc["Slytherin", "b"]) * 100
	df_test.loc[:, "Gryffindor"] = calc_proba(normalized_val, df_result.loc["Gryffindor", :"Flying"], df_result.loc["Gryffindor", "b"]) * 100
	df_test.loc[:, "Hufflepuff"] = calc_proba(normalized_val, df_result.loc["Hufflepuff", :"Flying"], df_result.loc["Hufflepuff", "b"]) * 100
	df_test.loc[:, "Ravenclaw"] = calc_proba(normalized_val, df_result.loc["Ravenclaw", :"Flying"], df_result.loc["Ravenclaw", "b"]) * 100
	return df_test

def create_house_sheet(df_array):
	print(df_array.iloc[:, -4:])
	last_columns = df_array.iloc[:, -4:]
	nameHouses = last_columns.idxmax(axis=1)
	print(nameHouses)
	df_array["Hogwarts House"] = nameHouses
	df_finals = pd.DataFrame(df_array.iloc[:, -1:])
	# df_array.to_csv("result.csv")
	# df_finals.to_csv("houses.csv")
	print(df_array)
	return df_array, df_finals

def main():
	# try:
		assert len(argv) == 3, "You need to pass your data test as argument and the trained value"
		df_toTest = pd.read_csv(argv[1], index_col = "Index")
		assert df_toTest is not None, "There is a problem with the dataset..."
		df_result = pd.read_csv(argv[2], index_col = "Index")
		assert df_toTest is not None, "There is a problem with the dataset..."
		X_toTest = df_toTest.drop(["Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand", "Astronomy"], axis=1, inplace=False)
		for i in range(len(X_toTest.columns)):
			X_toTest.loc[X_toTest.iloc[:,i].isna(), X_toTest.columns[i]] = df_result.loc["Mean", X_toTest.columns[i]]
		prob_array = create_prob_arr(X_toTest, df_result)
		df_array, df_finals = create_house_sheet(prob_array)
		df_finals.to_csv("houses.csv")
		df_array.to_csv("result.csv")

	# except Exception as e:
	# 	print(f'caught {type(e)}: e')
if __name__ == "__main__":
	main()
