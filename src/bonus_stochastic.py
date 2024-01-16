import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sys import argv
from logreg_train import describe_light, normalize_value


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
    y_slytherin, y_ravenclaw, y_gryffindor, y_hufflepuff = y
    w_slytherin, w_ravenclaw, w_gryffindor, w_hufflepuff = w
    b_slytherin, b_ravenclaw, b_gryffindor, b_hufflepuff = b
    cost_slytherin = 0
    cost_ravenclaw = 0
    cost_gryffindor = 0
    cost_hufflepuff = 0
    for i in range(m):
        z_slytherin = np.dot(w_slytherin, X.iloc[i].values) + b_slytherin
        f_wb_slytherin = 1 / (1 + np.exp(-z_slytherin))
        cost_slytherin += -y_slytherin[i] * np.log(f_wb_slytherin) - (1 - y_slytherin[i])*np.log(1-f_wb_slytherin)

        z_ravenclaw = np.dot(w_ravenclaw, X.iloc[i].values) + b_ravenclaw
        f_wb_ravenclaw = 1 / (1 + np.exp(-z_ravenclaw))
        cost_ravenclaw += -y_ravenclaw[i] * np.log(f_wb_ravenclaw) - (1 - y_ravenclaw[i])*np.log(1-f_wb_ravenclaw)

        z_gryffindor = np.dot(w_gryffindor, X.iloc[i].values) + b_gryffindor
        f_wb_gryffindor = 1 / (1 + np.exp(-z_gryffindor))
        cost_gryffindor += -y_gryffindor[i] * np.log(f_wb_gryffindor) - (1 - y_gryffindor[i])*np.log(1-f_wb_gryffindor)

        z_hufflepuff = np.dot(w_hufflepuff, X.iloc[i].values) + b_hufflepuff
        f_wb_hufflepuff = 1 / (1 + np.exp(-z_hufflepuff))
        cost_hufflepuff += -y_hufflepuff[i] * np.log(f_wb_hufflepuff) - (1 - y_hufflepuff[i])*np.log(1-f_wb_hufflepuff)

    cost_slytherin = cost_slytherin / m
    cost_ravenclaw = cost_ravenclaw / m
    cost_gryffindor = cost_gryffindor / m
    cost_hufflepuff = cost_hufflepuff / m

    reg_part_slytherin = 0
    reg_part_ravenclaw = 0
    reg_part_gryffindor = 0
    reg_part_hufflepuff = 0

    for i in range(n):
        reg_part_slytherin += w_slytherin[i]**2
        reg_part_ravenclaw += w_ravenclaw[i]**2
        reg_part_gryffindor += w_gryffindor[i]**2
        reg_part_hufflepuff += w_hufflepuff[i]**2

    cost_slytherin = cost_slytherin + (lambda_ / (2*m)) * reg_part_slytherin
    cost_ravenclaw = cost_ravenclaw + (lambda_ / (2*m)) * reg_part_ravenclaw
    cost_gryffindor = cost_gryffindor + (lambda_ / (2*m)) * reg_part_gryffindor
    cost_hufflepuff = cost_hufflepuff + (lambda_ / (2*m)) * reg_part_hufflepuff

    return (cost_slytherin, cost_ravenclaw, cost_gryffindor, cost_hufflepuff)


def updateWb(X, y, w, b, lambda_, alpha):
    n = len(X)
    m = 1
    dj_dw = np.array([np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)])
    dj_db = np.zeros(4)
    y_slytherin, y_ravenclaw, y_gryffindor, y_hufflepuff = y
    w_slytherin, w_ravenclaw, w_gryffindor, w_hufflepuff = w
    b_slytherin, b_ravenclaw, b_gryffindor, b_hufflepuff = b
    z_slytherin = np.dot(X, w_slytherin) + b_slytherin
    f_wb_i_slytherin = 1 / (1 + np.exp(-z_slytherin))
    err_i_slytherin  = f_wb_i_slytherin  - y_slytherin

    z_ravenclaw = np.dot(X, w_ravenclaw) + b_ravenclaw
    f_wb_i_ravenclaw = 1 / (1 + np.exp(-z_ravenclaw))
    err_i_ravenclaw  = f_wb_i_ravenclaw  - y_ravenclaw

    z_gryffindor = np.dot(X, w_gryffindor) + b_gryffindor
    f_wb_i_gryffindor = 1 / (1 + np.exp(-z_gryffindor))
    err_i_gryffindor  = f_wb_i_gryffindor  - y_gryffindor

    z_hufflepuff = np.dot(X, w_hufflepuff) + b_hufflepuff
    f_wb_i_hufflepuff = 1 / (1 + np.exp(-z_hufflepuff))
    err_i_hufflepuff = f_wb_i_hufflepuff  - y_hufflepuff
    for j in range(n):
        dj_dw[0][j] = dj_dw[0][j] + err_i_slytherin * X[j]
        dj_dw[1][j] = dj_dw[1][j] + err_i_ravenclaw * X[j]
        dj_dw[2][j] = dj_dw[2][j] + err_i_gryffindor * X[j]
        dj_dw[3][j] = dj_dw[3][j] + err_i_hufflepuff * X[j]
        dj_db[0] = dj_db[0] + err_i_slytherin
        dj_db[1] = dj_db[1] + err_i_ravenclaw
        dj_db[2] = dj_db[2] + err_i_gryffindor
        dj_db[3] = dj_db[3] + err_i_hufflepuff
    dj_dw =  dj_dw/m
    dj_db = dj_db/m

    for j in range(n):
        dj_dw[0][j] = dj_dw[0][j] + (lambda_/m) * w_slytherin[j]
        dj_dw[1][j] = dj_dw[1][j] + (lambda_/m) * w_ravenclaw[j]
        dj_dw[2][j] = dj_dw[2][j] + (lambda_/m) * w_gryffindor[j]
        dj_dw[3][j] = dj_dw[3][j] + (lambda_/m) * w_hufflepuff[j]

    W_updated = w - alpha * dj_dw
    b_updated = b - alpha * dj_db
    return (W_updated, b_updated)

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
        result[0][k],result[1][k], result[2][k], result[3][k] = (computeCost(X, y, w, b, lambda_))
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
