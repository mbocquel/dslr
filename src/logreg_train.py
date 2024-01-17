import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sys import argv


def describe_light(df):
    """
    Program that describe a dataset.
    """
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
        stats.loc['Mean', col_name[i]] = mean
        stats.loc['Std', col_name[i]] = std
    return stats


def normalize_value(df):
    dfNum = df.select_dtypes(include=['int64','float64'])
    stats = describe_light(dfNum)
    dfN = df.copy()
    for i in range(len(dfNum.columns)):
        mean = stats.loc["Mean", dfNum.columns[i]]
        std = stats.loc["Std", dfNum.columns[i]]
        dfN.loc[:, dfNum.columns[i]] = (dfN.loc[:, dfNum.columns[i]] - mean) / std
    return dfN


def computeCost(X, y, w, b, lambda_ = 1):
    """
    X (ndarray (m,n): Data, m examples with n features
    y (ndarray (m,)): target values
    w (ndarray (n,)): model parameters
    b (scalar)      : model parameter
    lambda_ (scalar): Controls amount of regularization
    Returns:
        cost (scalar):  cost
    """
    m, n = X.shape
    Z = X @ w + b
    F_wb = 1 / (1 + np.exp(-Z))
    cost = -np.sum(y * np.log(F_wb) + (1 - y) * np.log(1 - F_wb))
    cost = cost / m + (lambda_ / (2*m)) * np.sum(w**2)
    return (cost)


def updateWb(X, y, w, b, lambda_, alpha):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.

    Z = np.dot(X.values, w) + b
    F_wb = 1 / (1 + np.exp(-Z))
    Err = F_wb - y

    dj_dw = np.dot(Err.T, X) / m
    dj_db = np.sum(Err) / m
    
    # Ajout de la régularisation
    dj_dw += (lambda_ / m) * w

    # Mise à jour des poids et biais
    w_updated = w - alpha * dj_dw
    b_updated = b - alpha * dj_db
    
    return w_updated, b_updated


def executeGradientDescentAlgo(X, y, alpha, lambda_, nb_iterations):
    m, n = X.shape
    w = np.array([np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)])
    b = np.zeros(4)
    result = np.array([np.zeros(nb_iterations), np.zeros(nb_iterations), np.zeros(nb_iterations), np.zeros(nb_iterations)])
    for i in tqdm(range(nb_iterations)):
        for j in range(4):
            w[j], b[j] = updateWb(X, y[j], w[j], b[j], lambda_, alpha)
            result[j][i] = computeCost(X, y[j], w[j], b[j], lambda_)
    return(w, b, result)


def logreg(df, alpha, lambda_, nb_iterations):
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
    return (executeGradientDescentAlgo(X, y, alpha, lambda_, nb_iterations))


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
        w, b, result = logreg(df_Normilised, alpha, lambda_, nb_iter)

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
        params.to_csv("params.csv", index=True)
        print("Results saves in params.csv")
    except Exception as err:
        print("Error: ", err)

if __name__ == "__main__":
    main()
