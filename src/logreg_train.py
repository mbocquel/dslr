import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sys import argv


def describe(df):
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
    stats = describe(dfNum)
    dfN = df.copy()
    for i in range(len(dfNum.columns)):
        mean = stats.loc["Mean", dfNum.columns[i]]
        std = stats.loc["Std", dfNum.columns[i]]
        dfN.loc[:, dfNum.columns[i]] = (dfN.loc[:, dfNum.columns[i]] - mean) / std
    return dfN


def computeCost(X, y, W, b, lambda_ = 1):
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
    cost = 0
    for i in range(m):
        z = np.dot(W, X.iloc[i].values) + b
        f_wb = 1 / (1 + np.exp(-z))
        cost += -y.iloc[i].item() * np.log(f_wb) - (1 - y.iloc[i].item())*np.log(1-f_wb)
    cost = cost / m
    reg_part = 0
    for i in range(n):
        reg_part += W[i]**2
    cost = cost + (lambda_ / (2*m)) * reg_part
    return cost


def updateWb(X, y, w, b, lambda_, alpha):
    m, n = X.shape
    dj_dw = [np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)]
    dj_db = [0, 0, 0, 0]
    # dj_dw = pd.DataFrame(columns=X.columns,
    #                  index=["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"], data=0.0)
    # dj_db = pd.DataFrame(columns=["b"],
    #                  index=["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"], data=0.0)
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
        col_name = X.columns[j]
        dj_dw.loc["Slytherin", col_name] = dj_dw.loc["Slytherin", col_name].item() + (lambda_/m) * w.loc["Slytherin", col_name].item()
        dj_dw.loc["Ravenclaw", col_name] = dj_dw.loc["Ravenclaw", col_name].item() + (lambda_/m) * w.loc["Ravenclaw", col_name].item()
        dj_dw.loc["Gryffindor", col_name] = dj_dw.loc["Gryffindor", col_name].item() + (lambda_/m) * w.loc["Gryffindor", col_name].item()
        dj_dw.loc["Hufflepuff", col_name] = dj_dw.loc["Hufflepuff", col_name].item() + (lambda_/m) * w.loc["Hufflepuff", col_name].item()

    W_updated = w - alpha * dj_dw
    b_updated = b - alpha * dj_db
    return (W_updated, b_updated)


def executeGradientDescentAlgo(X, y, alpha, lambda_, nb_iterations):
    m, n = X.shape
    w = [np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)]
    # w  = pd.DataFrame(columns=X.columns,
    #                  index=["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"], data=0)
    #
    b = [0, 0, 0, 0]
    # b = pd.DataFrame(columns=["b"],
    #                  index=["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"], data=0)
    # result = pd.DataFrame(columns=["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"])
    for i in tqdm(range(nb_iterations)):
        w, b = updateWb(X, y, w, b, lambda_, alpha)
        # result.append(computeCost(X, y, w, b, lambda_))
    return(w, b)


def logreg(df):
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
    y = [slytherin_df.iloc[:, len(slytherin_df.columns)-1:].to_numpy(),
         ravenclaw_df.iloc[:, len(ravenclaw_df.columns)-1:].to_numpy(),
         gryffindor_df.iloc[:, len(gryffindor_df.columns)-1:].to_numpy(),
         hufflepuff_df.iloc[:, len(hufflepuff_df.columns)-1:].to_numpy()]
    # y_slytherin = slytherin_df.iloc[:, len(slytherin_df.columns)-1:]
    # y_ravenclaw = ravenclaw_df.iloc[:, len(ravenclaw_df.columns)-1:]
    # y_gryffindor = gryffindor_df.iloc[:, len(gryffindor_df.columns)-1:]
    # y_hufflepuff = hufflepuff_df.iloc[:, len(hufflepuff_df.columns)-1:]
    alpha = 0.3
    lambda_ = 1
    nb_iterations = 150
    return (executeGradientDescentAlgo(X, y, alpha, lambda_, nb_iterations))


def plot_algo_convergence(result_sly, result_rav, result_gryf, result_huf):
    x = range(len(result_sly))
    plt.plot(x, result_sly, 'g', label = "Slytherin")
    plt.plot(x, result_rav, 'b', label = "Ravenclaw")
    plt.plot(x, result_gryf, 'r', label = "Gryffindor")
    plt.plot(x, result_huf, 'y', label = "Hufflepuff")
    plt.xlabel("Algo iterations")
    plt.ylabel("Cost")
    plt.title("Evolution of the Cost with gradient descent iterations")
    plt.legend()
    plt.show()


def main():
    try:
        assert len(argv) == 2, "You need to pass your data file as argument"
        df = pd.read_csv(argv[1], index_col = "Index")
        assert df is not None, "There is a problem with the dataset..."
        df_bis = df.drop(['First Name', 'Last Name', "Birthday", "Best Hand"], axis=1, inplace=False)
        stats = describe(df_bis)

        ## on remplace les trous de donnes par le mean de la ligne
        for i in range(1, len(df_bis.columns)):
            df_bis.loc[df_bis.iloc[:,i].isna(), df_bis.columns[i]] = stats.loc["Mean", df_bis.columns[i]]

        ## on Normalise les donnees
        df_Normilised = normalize_value(df_bis)
        df_Normilised.drop("Astronomy", axis=1, inplace=True)

        # On execute le resultat de l'algo

        w, b = logreg(df_Normilised.copy())
        # print("...done")
        # print("Computing for Ravenclaw ...")
        # w_rav, b_rav, result_rav= logreg(df_Normilised.copy(), "Ravenclaw")
        # print("...done")
        # print("Computing for Gryffindor ...")
        # w_gryf, b_gryf, result_gryf = logreg(df_Normilised.copy(), "Gryffindor")
        # print("...done")
        # print("Computing for Hufflepuff ...")
        # w_huf, b_huf, result_huf = logreg(df_Normilised.copy(), "Hufflepuff")
        # print("...done")

        # Plot convergence
        # plot_algo_convergence(result_sly, result_rav, result_gryf, result_huf)
        print(w)
        print(b)
        # Enregistrement
        # params = stats.copy()
        # params.drop("Astronomy", axis=1, inplace=True)
        # params.loc["Slytherin", :] = w_sly
        # params.loc["Ravenclaw", :] = w_rav
        # params.loc["Gryffindor", :] = w_gryf
        # params.loc["Hufflepuff", :] = w_huf
        # params.loc["Slytherin", "b"] = b_sly
        # params.loc["Ravenclaw", "b"] = b_rav
        # params.loc["Gryffindor", "b"] = b_gryf
        # params.loc["Hufflepuff", "b"] = b_huf
        # params.to_csv("params.csv", index=True)
        # print(params)
        return 0
    except AssertionError as msg:
        print("AssertionError:", msg)
        return 1
    except Exception as err:
        print("Error: ", err)
        return 1

if __name__ == "__main__":
    main()
