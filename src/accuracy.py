import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def main():
	df_true = pd.read_csv("dataset_truth.csv", index_col = "Index")
	df_esti = pd.read_csv("houses.csv", index_col = "Index")
	print(df_esti.loc[:, "Hogwarts House":])
	print(df_true.loc[:, "Hogwarts House":])
	print("accuracy score",accuracy_score(df_true,df_esti))

if __name__ == "__main__":
	main()
