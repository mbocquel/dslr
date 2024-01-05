import pandas as pd
from sys import argv


def main():
    """
    Program that uses Machine Learning to find the corect
    parameters for a linear regression.
    """
    try:
        assert len(argv) == 2, "You need to pass your data file as argument"
        df = pd.read_csv(argv[1], index_col = "Index")
        assert df is not None, "There is a problem with the dataset..."
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
            quartile25 = float(sorted(col_allNum)[int(count/4)])
            quartile50 = float(sorted(col_allNum)[int(count/2)])
            quartile75 = float(sorted(col_allNum)[int(3 * count/4)])
            maxval = sorted(col_allNum)[count - 1]
            minval = sorted(col_allNum)[0]
            stats.loc['Count', col_name[i]] = count
            stats.loc['Mean', col_name[i]] = mean
            stats.loc['Std', col_name[i]] = std
            stats.loc['Min', col_name[i]] = minval
            stats.loc['25%', col_name[i]] = quartile25
            stats.loc['50%', col_name[i]] = quartile50
            stats.loc['75%', col_name[i]] = quartile75
            stats.loc['Max', col_name[i]] = maxval
        print(stats)
        return 0
    except AssertionError as msg:
        print("AssertionError:", msg)
        return 1
    except Exception as err:
        print("Error: ", err)
        return 1


if __name__ == "__main__":
    main()
