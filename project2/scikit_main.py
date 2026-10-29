from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from load_data import load
import matplotlib.pyplot as plt


from plotter import plot_regression, plot_residuals


def main(csv_name):
    x, y = load(csv_name)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=67)

    reg = LinearRegression().fit(x_train, y_train)

    print("R^2 on the data set: ", reg.score(x_test, y_test))
    plot_regression(reg, x, y)
    plot_residuals(reg, x, y)


main("project2\\Data\\2D L.csv")




