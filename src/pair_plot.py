import pandas as pd
from sys import argv
import matplotlib.pyplot as plt


def main():
    """
    Program that shows pair plots for our dataset
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
        nb_col_graph = len(col_names)
        nb_lignes_graph = len(col_names)
        plt.rcParams.update({'font.size': 7})
        fig, axs = plt.subplots(nb_lignes_graph, nb_col_graph)
        for i in range(len(col_names)):
            for j in range(len(col_names)):
                if (i == j):
                    axs[i, j].hist(Ravenclaw.loc[:,col_names[i]], alpha = 0.5, lw=3, label="Ravenclaw", color="b")
                    axs[i, j].hist(Hufflepuff.loc[:,col_names[i]], alpha = 0.5, lw=3, label="Hufflepuff", color="y")
                    axs[i, j].hist(Slytherin.loc[:,col_names[i]], alpha = 0.5, lw=3, label="Slytherin", color="g")
                    axs[i, j].hist(Gryffindor.loc[:,col_names[i]], alpha = 0.5, lw=3, label="Gryffindor", color="r")
                    axs[i, j].set(xlabel=col_names[j], ylabel=(col_names[i])[:10])
                else:
                    axs[i, j].plot(Ravenclaw.loc[:,col_names[j]], Ravenclaw.loc[:,col_names[i]], '.',  label="Ravenclaw", color="b", markersize=1)
                    axs[i, j].plot(Hufflepuff.loc[:,col_names[j]], Hufflepuff.loc[:,col_names[i]], '.',  label="Hufflepuff", color="y", markersize=1)
                    axs[i, j].plot(Slytherin.loc[:,col_names[j]], Slytherin.loc[:,col_names[i]], '.',  label="Slytherin", color="g", markersize=1)
                    axs[i, j].plot(Gryffindor.loc[:,col_names[j]], Gryffindor.loc[:,col_names[i]], '.',  label="Gryffindor", color="r", markersize=1)
                    axs[i, j].set(xlabel=col_names[j], ylabel=(col_names[i])[:10])
        for ax in axs.flat:
            ax.label_outer()
        plt.subplots_adjust(wspace=0.2, hspace=0.2)    
        plt.legend(['Ravenclaw', 'Hufflepuff', 'Slytherin', 'Gryffindor'], loc='upper center', bbox_to_anchor=(-0.9, -0.8), fancybox=True, shadow=True, ncol=2)
        manager = plt.get_current_fig_manager()
        manager.set_window_title("Pair plot")
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
