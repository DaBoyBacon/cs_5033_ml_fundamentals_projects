import numpy as np

def load(csv_file: str):

    data = np.genfromtxt(csv_file, delimiter=",")

    X = data[:, :-1]   # all rows, all columns but the last
    y = data[:, -1]    # all rows, last column only
    return X, y



if __name__ == "__main__":
    import glob
    for csv_file in sorted(glob.glob("Data/*.csv")):
        X, y = load(csv_file)
        print(csv_file, "X:", X.shape, "y:", y.shape)