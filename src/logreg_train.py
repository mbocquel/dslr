import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    for i in range(m):
        z = np.dot(X.iloc[i].values, w) + b
        f_wb_i = 1 / (1 + np.exp(-z))
        err_i  = f_wb_i  - y.iloc[i].item()
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X.iloc[i,j].item()
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]
    W_updated = w - alpha * dj_dw
    b_updated = b - alpha * dj_db
    return (W_updated, b_updated)


def executeGradientDescentAlgo(X, y, alpha, lambda_, nb_iter):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    result = []
    for i in range(nb_iter):
        w, b = updateWb(X, y, w, b, lambda_, alpha)
        result.append(computeCost(X, y, w, b, lambda_))
    return(w, b, result)


def logreg(df, house_name):
    df.loc[:,"Hogwarts House"] == house_name
    # df.loc[df.loc[:,"Hogwarts House"] == house_name, :]
    df.loc[df.loc[:,"Hogwarts House"] == house_name, "Test" + house_name] = 1
    df.loc[~(df.loc[:,"Hogwarts House"] == house_name), "Test" + house_name] = 0
    df.drop("Hogwarts House", axis=1, inplace=True)
    # df.drop("Astronomy", axis=1, inplace=True)
    for i in range(len(df.columns)):
        df = df.loc[~df.isna()[df.columns[i]], :]
    X = df.iloc[:, :len(df.columns)-1]
    y = df.iloc[:, len(df.columns)-1:]
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
        print("Computing for Slytherin ...")
        w_sly, b_sly, result_sly = logreg(df_Normilised.copy(), "Slytherin")
        print("...done")
        print("Computing for Ravenclaw ...")
        w_rav, b_rav, result_rav= logreg(df_Normilised.copy(), "Ravenclaw")
        print("...done")
        print("Computing for Gryffindor ...")
        w_gryf, b_gryf, result_gryf = logreg(df_Normilised.copy(), "Gryffindor")
        print("...done")
        print("Computing for Hufflepuff ...")
        w_huf, b_huf, result_huf = logreg(df_Normilised.copy(), "Hufflepuff")
        print("...done")

        # Plot convergence
        plot_algo_convergence(result_sly, result_rav, result_gryf, result_huf)

        # Enregistrement
        params = stats.copy()
        params.drop("Astronomy", axis=1, inplace=True)
        params.loc["Slytherin", :] = w_sly
        params.loc["Ravenclaw", :] = w_rav
        params.loc["Gryffindor", :] = w_gryf
        params.loc["Hufflepuff", :] = w_huf
        params.loc["Slytherin", "b"] = b_sly
        params.loc["Ravenclaw", "b"] = b_rav
        params.loc["Gryffindor", "b"] = b_gryf
        params.loc["Hufflepuff", "b"] = b_huf
        params.to_csv("params.csv", index=False)
        print(params)
        return 0
    except AssertionError as msg:
        print("AssertionError:", msg)
        return 1
    except Exception as err:
        print("Error: ", err)
        return 1

if __name__ == "__main__":
    main()
