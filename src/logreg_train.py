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


def updateWb(X_slytherin, y_slytherin,
            X_ravenclaw, y_ravenclaw,
            X_gryffindor, y_gryffindor,
            X_hufflepuff, y_hufflepuff,
            w, b, lambda_, alpha):
    m, n = X_slytherin.shape
    dj_dw = pd.DataFrame(columns=X_slytherin.columns,
                     index=["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"], data=0)
    dj_db = pd.DataFrame(columns=["b"],
                     index=["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"], data=0)
    for i in range(m):
        z_slytherin = np.dot(X_slytherin.iloc[i].values, w.loc["Slytherin", :].values) + b.loc["Slytherin", "b"].item()
        f_wb_i_slytherin = 1 / (1 + np.exp(-z_slytherin))
        err_i_slytherin  = f_wb_i_slytherin  - y_slytherin.iloc[i].item()

        z_ravenclaw = np.dot(X_ravenclaw.iloc[i].values, w.loc["Ravenclaw", :].values) + b.loc["Ravenclaw", "b"].item()
        f_wb_i_ravenclaw = 1 / (1 + np.exp(-z_ravenclaw))
        err_i_ravenclaw  = f_wb_i_ravenclaw  - y_ravenclaw.iloc[i].item()
        
        z_gryffindor = np.dot(X_gryffindor.iloc[i].values, w.loc["Gryffindor", :].values) + b.loc["Gryffindor", "b"].item()
        f_wb_i_gryffindor = 1 / (1 + np.exp(-z_gryffindor))
        err_i_gryffindor  = f_wb_i_gryffindor  - y_gryffindor.iloc[i].item()

        z_hufflepuff = np.dot(X_hufflepuff.iloc[i].values, w.loc["Hufflepuff", :].values) + b.loc["Hufflepuff", "b"].item()
        f_wb_i_hufflepuff = 1 / (1 + np.exp(-z_hufflepuff))
        err_i_hufflepuff = f_wb_i_hufflepuff  - y_hufflepuff.iloc[i].item()

        # f_wb_i = 1 / (1 + np.exp(-z))
        # err_i  = f_wb_i  - y.iloc[i].item()
        for j in range(n):
            col_name = X_slytherin.columns[j]
            dj_dw.loc["Slytherin", col_name] = dj_dw.loc["Slytherin", col_name] + err_i_slytherin * X_slytherin.iloc[i,j].item()
            dj_dw.loc["Ravenclaw", col_name] = dj_dw.loc["Ravenclaw", col_name] + err_i_ravenclaw * X_ravenclaw.iloc[i,j].item()
            dj_dw.loc["Gryffindor", col_name] = dj_dw.loc["Gryffindor", col_name] + err_i_gryffindor * X_gryffindor.iloc[i,j].item()
            dj_dw.loc["Hufflepuff", col_name] = dj_dw.loc["Hufflepuff", col_name] + err_i_hufflepuff * X_hufflepuff.iloc[i,j].item()
        
        dj_db.loc["Slytherin", "b"] = dj_db.loc["Slytherin", "b"] + err_i_slytherin
        dj_db.loc["Ravenclaw", "b"] = dj_db.loc["Ravenclaw", "b"] + err_i_ravenclaw
        dj_db.loc["Gryffindor", "b"] = dj_db.loc["Gryffindor", "b"] + err_i_gryffindor
        dj_db.loc["Hufflepuff", "b"] = dj_db.loc["Hufflepuff", "b"] + err_i_hufflepuff
        
    dj_dw =  dj_dw/m
    dj_db = dj_db/m

    for j in range(n):
        col_name = X_slytherin.columns[j]
        dj_dw.loc["Slytherin", col_name] = dj_dw.loc["Slytherin", col_name] + (lambda_/m) * w.loc["Slytherin", col_name]
        dj_dw.loc["Ravenclaw", col_name] = dj_dw.loc["Ravenclaw", col_name] + (lambda_/m) * w.loc["Ravenclaw", col_name]
        dj_dw.loc["Gryffindor", col_name] = dj_dw.loc["Gryffindor", col_name] + (lambda_/m) * w.loc["Gryffindor", col_name]
        dj_dw.loc["Hufflepuff", col_name] = dj_dw.loc["Hufflepuff", col_name] + (lambda_/m) * w.loc["Hufflepuff", col_name]
    
    W_updated = w - alpha * dj_dw
    b_updated = b - alpha * dj_db
    return (W_updated, b_updated)


def executeGradientDescentAlgo(X_slytherin, y_slytherin, 
                                X_ravenclaw, y_ravenclaw, 
                                X_gryffindor, y_gryffindor,
                                X_hufflepuff, y_hufflepuff,
                                alpha, lambda_, nb_iterations):
    w = pd.DataFrame(columns=X_slytherin.columns,
                     index=["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"], data=0)
    b = pd.DataFrame(columns=["b"],
                     index=["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"], data=0)
    result = pd.DataFrame(columns=["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"])
    for i in tqdm(range(nb_iterations)):
        w, b = updateWb(X_slytherin, y_slytherin,
                        X_ravenclaw, y_ravenclaw,
                        X_gryffindor, y_gryffindor,
                        X_hufflepuff, y_hufflepuff,
                        w, b, lambda_, alpha)
        # result.append(computeCost(X, y, w, b, lambda_))
    return(w, b)


def logreg(df, house_name):
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

    X_slytherin = slytherin_df.iloc[:, :len(slytherin_df.columns)-1]
    y_slytherin = slytherin_df.iloc[:, len(slytherin_df.columns)-1:]
    X_ravenclaw = ravenclaw_df.iloc[:, :len(ravenclaw_df.columns)-1]
    y_ravenclaw = ravenclaw_df.iloc[:, len(ravenclaw_df.columns)-1:]
    X_gryffindor = gryffindor_df.iloc[:, :len(gryffindor_df.columns)-1]
    y_gryffindor = gryffindor_df.iloc[:, len(gryffindor_df.columns)-1:]
    X_hufflepuff = hufflepuff_df.iloc[:, :len(hufflepuff_df.columns)-1]
    y_hufflepuff = hufflepuff_df.iloc[:, len(hufflepuff_df.columns)-1:]
    alpha = 0.3
    lambda_ = 1
    nb_iterations = 150
    return (executeGradientDescentAlgo(X_slytherin, y_slytherin, 
                                       X_ravenclaw, y_ravenclaw, 
                                       X_gryffindor, y_gryffindor, 
                                       X_hufflepuff, y_hufflepuff,
                                       alpha, lambda_, nb_iterations))


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
        params.to_csv("params.csv", index=True)
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
