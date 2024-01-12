import pandas as pd
from sklearn.metrics import accuracy_score
from sys import argv


def main():
    try:
        assert len(argv) == 3, "You need to pass the path of the Truth file and of the prediction file"
        df_true = pd.read_csv(argv[1], index_col = "Index")
        df_esti = pd.read_csv(argv[2], index_col = "Index")
        print(f'Accuracy score {accuracy_score(df_true,df_esti) * 100}%' )
    except Exception as e:
        print(f'Error: {e}')


if __name__ == "__main__":
	main()
