import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from logreg_train import normalize_value, describe_light, computeCost, updateWb
from sys import argv


def executeGradientDescentAlgo_miniBatch(X, y, mini_batchesX, mini_batchY, alpha, lambda_, nb_iterations):
    m, n = X.shape
    w = np.array([np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)])
    b = np.zeros(4)
    result = np.array([np.zeros(nb_iterations), np.zeros(nb_iterations), np.zeros(nb_iterations), np.zeros(nb_iterations)])
    l = 0
    for i in tqdm(range(nb_iterations)):
        for k in range(4):
            for j in range(len(mini_batchesX)):
                w[k], b[k] = updateWb(mini_batchesX[j], mini_batchY[j][k], w[k], b[k], lambda_, alpha)
            result[k] = computeCost(X, y[k], w[k], b[k], lambda_)
        l += 1
    return(w, b, result)


def logreg_minibatch(df, alpha, lambda_, nb_iterations, nb_batch):
    # On supprime les valeurs NaN
    for i in range(len(df.columns)):
        df = df.loc[~df.isna()[df.columns[i]], :]

    # On ajoute les test
    slytherin_df = df.copy()
    slytherin_df.loc[:, "Test Slytherin"] = 0
    slytherin_df.loc[df.loc[:,"Hogwarts House"] == "Slytherin", "Test Slytherin"] = 1
    ravenclaw_df= df.copy()
    ravenclaw_df.loc[:, "Test Ravenclaw"] = 0
    ravenclaw_df.loc[df.loc[:,"Hogwarts House"] == "Ravenclaw", "Test Ravenclaw"] = 1
    gryffindor_df= df.copy()
    gryffindor_df.loc[:, "Test Gryffindor"] = 0
    gryffindor_df.loc[df.loc[:,"Hogwarts House"] == "Gryffindor", "Test Gryffindor"] = 1
    hufflepuff_df= df.copy()
    hufflepuff_df.loc[:, "Test Hufflepuff"] = 0
    hufflepuff_df.loc[df.loc[:,"Hogwarts House"] == "Hufflepuff", "Test Hufflepuff"] = 1
    slytherin_df.drop("Hogwarts House", axis=1, inplace=True)
    ravenclaw_df.drop("Hogwarts House", axis=1, inplace=True)
    gryffindor_df.drop("Hogwarts House", axis=1, inplace=True)
    hufflepuff_df.drop("Hogwarts House", axis=1, inplace=True)

    X = slytherin_df.iloc[:, :len(slytherin_df.columns)-1]
    y = np.array([slytherin_df.iloc[:, len(slytherin_df.columns)-1:].to_numpy()[:, 0],
         ravenclaw_df.iloc[:, len(ravenclaw_df.columns)-1:].to_numpy()[:, 0],
         gryffindor_df.iloc[:, len(gryffindor_df.columns)-1:].to_numpy()[:, 0],
         hufflepuff_df.iloc[:, len(hufflepuff_df.columns)-1:].to_numpy()[:, 0]])
    m, n = X.shape
    batch_size = int (m/nb_batch)
    mini_batchesX = [X.iloc[i:i+batch_size] for i in range(0, len(X), batch_size)]
    mini_batchY_slytherin = [y[0][i:i + batch_size] for i in range(0, len(y[0]), batch_size)]
    mini_batchY_ravenclaw = [y[1][i:i + batch_size] for i in range(0, len(y[1]), batch_size)]
    mini_batchY_gryffindor = [y[2][i:i + batch_size] for i in range(0, len(y[2]), batch_size)]
    mini_batchY_hufflepuff = [y[3][i:i + batch_size] for i in range(0, len(y[3]), batch_size)]
    mini_batchY = [[mini_batchY_slytherin[i], mini_batchY_ravenclaw[i], mini_batchY_gryffindor[i],  mini_batchY_hufflepuff[i]]
               for i in range(len(mini_batchY_slytherin))]
    return (executeGradientDescentAlgo_miniBatch(X, y, mini_batchesX, mini_batchY, alpha, lambda_, nb_iterations))


def processInputs():
    datasetPath = ""
    alpha = 0.5
    lambda_ = 0.0
    nb_iter = 20
    nb_batch = 20
    try:
        if (len(argv) >= 2):
            datasetPath = argv[1]
        if (len(argv) >= 3):
            alpha = float(argv[2])
        if (len(argv) >= 4):
            lambda_ = float(argv[3])
        if (len(argv) >= 5):
            nb_iter = int(argv[4])
        if (len(argv) == 6):
            nb_batch = int(argv[5])
        return (datasetPath, alpha, lambda_, nb_iter, nb_batch)
    except Exception:
        return (None, None, None, None)


def main():
    try:
        assert len(argv) >= 2, "Not enough arguments. \nHelp : datasetfile alpha lambda nb_iteration"
        assert len(argv) <= 6 , "Too many arguments"
        datasetPath, alpha, lambda_, nb_iter, nb_batch = processInputs()
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
        print(f'Computing with alpha = {str(alpha)}, lambda_ = {str(lambda_)}, {str(nb_iter)} iterations and {str(nb_batch)} batchs')
        w, b, result = logreg_minibatch(df_Normilised, alpha, lambda_, nb_iter, nb_batch)

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
        params.to_csv("params_minibatch.csv", index=True)
        print("Results saves in params_minibatch.csv")
    except Exception as err:
        print("Error: ", err)

if __name__ == "__main__":
    main()
