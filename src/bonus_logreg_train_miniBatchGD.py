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
    m, n = X.shape
    dj_dw = np.array([np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)])
    dj_db = np.zeros(4)
    y_slytherin, y_ravenclaw, y_gryffindor, y_hufflepuff = y
    w_slytherin, w_ravenclaw, w_gryffindor, w_hufflepuff = w
    b_slytherin, b_ravenclaw, b_gryffindor, b_hufflepuff = b
    for i in range(m):
        z_slytherin = np.dot(X.iloc[i].values, w_slytherin) + b_slytherin
        f_wb_i_slytherin = 1 / (1 + np.exp(-z_slytherin))
        err_i_slytherin  = f_wb_i_slytherin  - y_slytherin[i]

        z_ravenclaw = np.dot(X.iloc[i].values, w_ravenclaw) + b_ravenclaw
        f_wb_i_ravenclaw = 1 / (1 + np.exp(-z_ravenclaw))
        err_i_ravenclaw  = f_wb_i_ravenclaw  - y_ravenclaw[i]
        
        z_gryffindor = np.dot(X.iloc[i].values, w_gryffindor) + b_gryffindor
        f_wb_i_gryffindor = 1 / (1 + np.exp(-z_gryffindor))
        err_i_gryffindor  = f_wb_i_gryffindor  - y_gryffindor[i]

        z_hufflepuff = np.dot(X.iloc[i].values, w_hufflepuff) + b_hufflepuff
        f_wb_i_hufflepuff = 1 / (1 + np.exp(-z_hufflepuff))
        err_i_hufflepuff = f_wb_i_hufflepuff  - y_hufflepuff[i]

        for j in range(n):
            dj_dw[0][j] = dj_dw[0][j] + err_i_slytherin * X.iloc[i,j].item()
            dj_dw[1][j] = dj_dw[1][j] + err_i_ravenclaw * X.iloc[i,j].item()
            dj_dw[2][j] = dj_dw[2][j] + err_i_gryffindor * X.iloc[i,j].item()
            dj_dw[3][j] = dj_dw[3][j] + err_i_hufflepuff * X.iloc[i,j].item()
        
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


def executeGradientDescentAlgo_miniBatch(X, y, mini_batchesX, mini_batchY, alpha, lambda_, nb_iterations):
    m, n = X.shape
    w = np.array([np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)])
    b = np.zeros(4)
    result = np.array([np.zeros(nb_iterations), np.zeros(nb_iterations), np.zeros(nb_iterations), np.zeros(nb_iterations)])
    k = 0
    for i in tqdm(range(nb_iterations)):
        for j in range(len(mini_batchesX)):
            w, b = updateWb(mini_batchesX[j], mini_batchY[j], w, b, lambda_, alpha)
        result[0][k], result[1][k], result[2][k], result[3][k] = computeCost(X, y, w, b, lambda_)
        k += 1
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
        params.to_csv("params.csv", index=True)
        print("Results saves in params.csv")
    except Exception as err:
        print("Error: ", err)

if __name__ == "__main__":
    main()
