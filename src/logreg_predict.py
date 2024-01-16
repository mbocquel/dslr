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


def create_prob_arr(df_test, params):
	result = df_test.copy()
	normalized_val = normalize_value(result, params)
	result.loc[:, "Slytherin"] = calc_proba(normalized_val, params.iloc[2, :-1], params.iloc[2, -1]) * 100
	result.loc[:, "Ravenclaw"] = calc_proba(normalized_val, params.iloc[3, :-1], params.iloc[3, -1]) * 100
	result.loc[:, "Gryffindor"] = calc_proba(normalized_val, params.iloc[4, :-1], params.iloc[4, -1]) * 100
	result.loc[:, "Hufflepuff"] = calc_proba(normalized_val, params.iloc[5, :-1], params.iloc[5, -1]) * 100
	return result


def create_prob_arr(df_test, params):
	result = df_test.copy()
	normalized_val = normalize_value(result, params)
	result.loc[:, "Slytherin"] = calc_proba(normalized_val, params.iloc[2, :-1], params.iloc[2, -1]) * 100
	result.loc[:, "Ravenclaw"] = calc_proba(normalized_val, params.iloc[3, :-1], params.iloc[3, -1]) * 100
	result.loc[:, "Gryffindor"] = calc_proba(normalized_val, params.iloc[4, :-1], params.iloc[4, -1]) * 100
	result.loc[:, "Hufflepuff"] = calc_proba(normalized_val, params.iloc[5, :-1], params.iloc[5, -1]) * 100
	return result


def create_house_sheet(prob_array):
	df_array = prob_array.copy()
def create_house_sheet(prob_array):
	df_array = prob_array.copy()
	last_columns = df_array.iloc[:, -4:]
	nameHouses = last_columns.idxmax(axis=1)
	df_array["Hogwarts House"] = nameHouses
	df_finals = pd.DataFrame(df_array.iloc[:, -1:])
	return df_finals

	return df_finals


def main():
    try:
        assert len(argv) == 3, "You need to pass your data test as argument and the trained value"
        df_toTest = pd.read_csv(argv[1], index_col = "Index")
        assert df_toTest is not None, "There is a problem with the dataset..."
        params = pd.read_csv(argv[2], index_col = 0)
        assert params is not None, "There is a problem with the dataset..."

        col_to_del = [col for col in df_toTest.columns if col not in params.columns]
        X_toTest = df_toTest.drop(col_to_del, axis=1, inplace=False)

        # Replace missing values with the mean of the feature
        for i in range(len(X_toTest.columns)):
            X_toTest.loc[X_toTest.iloc[:,i].isna(), X_toTest.columns[i]] = params.loc["Mean", X_toTest.columns[i]]

        prob_array = create_prob_arr(X_toTest, params)
        df_finals = create_house_sheet(prob_array)
        df_finals.to_csv("houses.csv")

    except Exception as e:
        print(f'caught {type(e)}: e')


if __name__ == "__main__":
	main()
