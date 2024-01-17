import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sys import argv
from logreg_train import describe_light, normalize_value, computeCost

def updateWb(X, y, w, b, lambda_, alpha):
    n = len(X)
    dj_dw = np.zeros((4, n))
    dj_db = np.zeros(4)

    for i in range(4):  # Pour chaque classe
        z = np.dot(X, w[i]) + b[i]
        f_wb_i = 1 / (1 + np.exp(-z))
        err_i = f_wb_i - y[i]

        dj_dw[i] = np.dot(err_i, X)
        dj_db[i] = np.sum(err_i)

        # Ajout de la régularisation
        dj_dw[i] += (lambda_) * w[i]

    # Mise à jour des poids et biais
    w_updated = w - alpha * dj_dw
    b_updated = b - alpha * dj_db
    
    return w_updated, b_updated

def executeGradientDescentAlgo(X, y, nb_iterations, lambda_, alpha):

    m, n = X.shape
    w = np.array([np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)])
    b = np.zeros(4)
    print(nb_iterations)
    result = np.array([np.zeros(nb_iterations), np.zeros(nb_iterations), np.zeros(nb_iterations), np.zeros(nb_iterations)])
    k = 0
    for i in tqdm(range(nb_iterations)):
        random_index = np.random.choice(m)
        X_i = X.iloc[random_index, :].values
        y_i = np.array([y[0][random_index], y[1][random_index], y[2][random_index], y[3][random_index]])
        w, b = updateWb(X_i, y_i, w, b, lambda_, alpha)
        for j in range(4):
            result[j][k] = computeCost(X, y[j], w[j], b[j], lambda_)
        k += 1
    return (w, b, result)


def logreg_stochastic(df_Normilised, alpha, lambda_, nb_iterations):
    for i in range(len(df_Normilised.columns)):
        df_Normilised = df_Normilised.loc[~df_Normilised.isna()[df_Normilised.columns[i]], :]
    # On ajoute les test
    slytherin_df = df_Normilised.copy()
    slytherin_df.loc[:, "Test Slytherin"] = 0
    slytherin_df.loc[df_Normilised.loc[:,"Hogwarts House"] == "Slytherin", "Test Slytherin"] = 1

    ravenclaw_df= df_Normilised.copy()
    ravenclaw_df.loc[:, "Test Ravenclaw"] = 0
    ravenclaw_df.loc[df_Normilised.loc[:,"Hogwarts House"] == "Ravenclaw", "Test Ravenclaw"] = 1

    gryffindor_df= df_Normilised.copy()
    gryffindor_df.loc[:, "Test Gryffindor"] = 0
    gryffindor_df.loc[df_Normilised.loc[:,"Hogwarts House"] == "Gryffindor", "Test Gryffindor"] = 1
    hufflepuff_df= df_Normilised.copy()
    hufflepuff_df.loc[:, "Test Hufflepuff"] = 0
    hufflepuff_df.loc[df_Normilised.loc[:,"Hogwarts House"] == "Hufflepuff", "Test Hufflepuff"] = 1
    slytherin_df.drop("Hogwarts House", axis=1, inplace=True)
    ravenclaw_df.drop("Hogwarts House", axis=1, inplace=True)
    gryffindor_df.drop("Hogwarts House", axis=1, inplace=True)
    hufflepuff_df.drop("Hogwarts House", axis=1, inplace=True)
    X = slytherin_df.iloc[:, :len(slytherin_df.columns)-1]
    y = np.array([slytherin_df.iloc[:, len(slytherin_df.columns)-1:].to_numpy()[:, 0],
            ravenclaw_df.iloc[:, len(ravenclaw_df.columns)-1:].to_numpy()[:, 0],
            gryffindor_df.iloc[:, len(gryffindor_df.columns)-1:].to_numpy()[:, 0],
            hufflepuff_df.iloc[:, len(hufflepuff_df.columns)-1:].to_numpy()[:, 0]])
    return (executeGradientDescentAlgo(X, y,nb_iterations,  lambda_, alpha))


def processInputs():
    datasetPath = ""
    alpha = 0.5
    lambda_ = 0.0
    nb_iter = 250
    try:
        if (len(argv) >= 2):
            datasetPath = argv[1]
        if (len(argv) >= 3):
            alpha = float(argv[2])
        if (len(argv) >= 4):
            lambda_ = float(argv[3])
        if (len(argv) == 5):
            nb_iter = int(argv[4])
        return (datasetPath, alpha, lambda_, nb_iter)
    except Exception:
        return (None, None, None, None)


def main():
    try:
        assert len(argv) >= 2, "Not enough arguments. \nHelp : datasetfile alpha lambda nb_iteration"
        assert len(argv) <= 5 , "Too many arguments"
        datasetPath, alpha, lambda_, nb_iter = processInputs()
        assert datasetPath is not None, "Please enter valid arguments"
        df = pd.read_csv(datasetPath, index_col = "Index")
        assert df is not None, "There is a problem with the dataset..."

        #Delete useless colums
        col_to_delete = ["First Name", "Last Name", "Birthday", "Best Hand", "Astronomy",
                         "Arithmancy", "Care of Magical Creatures"]
        df.drop(col_to_delete, axis=1, inplace=True)

        #Savings the mean and std for Normalisation
        stats = describe_light(df)

        #Normalising the data
        df_Normilised = normalize_value(df)

        #Launching the algo
        print("Computing with alpha = " + str(alpha) + " lambda_ = " + str(lambda_) + " with " + str(nb_iter) + " iterations" )
        w, b, result = logreg_stochastic(df_Normilised, alpha, lambda_, nb_iter)

        #Saving the results
        params = stats.copy()
        params.loc["Slytherin", :] = w[0]
        params.loc["Ravenclaw", :] = w[1]
        params.loc["Gryffindor", :] = w[2]
        params.loc["Hufflepuff", :] = w[3]
        params.loc["Slytherin", "b"] = b[0]
        params.loc["Ravenclaw", "b"] = b[1]
        params.loc["Gryffindor", "b"] = b[2]
        params.loc["Hufflepuff", "b"] = b[3]
        params.to_csv("params_stochastic.csv", index=True)
        print("Results saves in params_stochastic.csv")
    except Exception as err:
        print("Error: ", err)


if (__name__ == "__main__"):
    main()
