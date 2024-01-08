import pandas as pd
from sys import argv
import matplotlib.pyplot as plt


def scatter(sujet, Ravenclaw, Hufflepuff, Slytherin, Gryffindor, col_names, nb_col_graph):
    print("     \033[32mShowing you the scatter plot for", sujet, "\033[0m")
    nb_lignes_graph = int((len(col_names)- 1)/nb_col_graph) + 1
    plt.figure(figsize=(nb_col_graph * 7, nb_lignes_graph * 7))
    j = 0
    for i in range(len(col_names)):
        if (col_names[i] != sujet):
            plt.subplot(nb_lignes_graph, nb_col_graph, j + 1)
            plt.title(sujet + " Vs " + col_names[i])
            plt.plot(Ravenclaw.loc[:,sujet], Ravenclaw.loc[:,col_names[i]], 'o',  label="Ravenclaw", color="b")
            plt.plot(Hufflepuff.loc[:,sujet], Hufflepuff.loc[:,col_names[i]], 'o',  label="Hufflepuff", color="y")
            plt.plot(Slytherin.loc[:,sujet], Slytherin.loc[:,col_names[i]], 'o',  label="Slytherin", color="g")
            plt.plot(Gryffindor.loc[:,sujet], Gryffindor.loc[:,col_names[i]], 'o',  label="Gryffindor", color="r")
            plt.xlabel(sujet)
            plt.ylabel(col_names[i])
            plt.legend()
            j += 1
    plt.show()


def showPossibilites(col_names):
    print("Possible features: ")
    for i in range(len(col_names)):
        print("     ",i + 1, "-", col_names[i])


def testParamsAreOk(value, col_names):
    try:
        result = int(value)
        assert result <= len(col_names)
        assert result >= 0
        return (result)
    except Exception:
        return (None)

def main():
    """
    Program that shows scatter_plot
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
        nb_col_graph = 5
        showPossibilites(col_names)
        userMainFeature = testParamsAreOk(input("Please enter the number of the feature to compare, or 0 to quit: "), col_names)
        while (1):
            if (userMainFeature is None):
                print("     \033[31mPlease enter a correct number\033[0m")
            elif (userMainFeature == 0):
                return 0
            else:
                scatter(col_names[userMainFeature - 1], Ravenclaw, Hufflepuff, Slytherin, Gryffindor, col_names, nb_col_graph)
            userMainFeature = testParamsAreOk(input("Please enter the number of the feature to compare, or 0 to quit: "), col_names)
    except AssertionError as msg:
        print("AssertionError:", msg)
        return 1
    except Exception as err:
        print("Error: ", err)
        return 1


if __name__ == "__main__":
    main()
