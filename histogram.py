import pandas as pd
from sys import argv
import matplotlib.pyplot as plt


def main():
    """
    Program that shows historgrams
    """
    try:
        assert len(argv) == 2, "You need to pass your data file as argument"
        df = pd.read_csv(argv[1], index_col = "Index")
        assert df is not None, "There is a problem with the dataset..."
        Ravenclaw = df[df.loc[:,"Hogwarts House"] == "Ravenclaw"].select_dtypes(include=['int64','float64'])
        Gryffindor = df[df.loc[:,"Hogwarts House"] == "Gryffindor"].select_dtypes(include=['int64','float64'])
        Slytherin = df[df.loc[:,"Hogwarts House"] == "Slytherin"].select_dtypes(include=['int64','float64'])
        Hufflepuff = df[df.loc[:,"Hogwarts House"] == "Hufflepuff"].select_dtypes(include=['int64','float64'])
        col_names = df.select_dtypes(include=['int64','float64']).columns
        nb_col_graph = 3
        nb_lignes_graph = int(len(col_names)/nb_col_graph) + 1
        plt.figure(figsize=(nb_col_graph * 5, nb_lignes_graph * 5))
        for i in range(len(col_names)):
            plt.subplot(nb_lignes_graph, nb_col_graph, i + 1)
            plt.title(col_names[i])
            plt.hist(Ravenclaw.loc[:,col_names[i]], alpha = 0.5, lw=3, label="Ravenclaw", color="b")
            plt.hist(Hufflepuff.loc[:,col_names[i]], alpha = 0.5, lw=3, label="Hufflepuff", color="y")
            plt.hist(Slytherin.loc[:,col_names[i]], alpha = 0.5, lw=3, label="Slytherin", color="g")
            plt.hist(Gryffindor.loc[:,col_names[i]], alpha = 0.5, lw=3, label="Gryffindor", color="r")
            plt.legend()
        plt.show()
        return 0
    except AssertionError as msg:
        print("AssertionError:", msg)
        return 1
    except Exception as err:
        print("Error: ", err)
        return 1


if __name__ == "__main__":
    main()
