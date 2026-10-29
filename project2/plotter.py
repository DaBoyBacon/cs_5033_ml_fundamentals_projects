import numpy as np
import matplotlib.pyplot as plt

def plot_regression(reg, X, y, names=None, target="y"):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[1]
    names = names or [f"x{i + 1}" for i in range(n)]

    if n == 1:
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], y, alpha=0.6, label="data")
        xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 100).reshape(-1, 1)
        ax.plot(xs, reg.predict(xs), color="red", label="fit")
        ax.set_xlabel(names[0])
        ax.set_ylabel(target)
        ax.legend()

    elif n == 2:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(X[:, 0], X[:, 1], y, alpha=0.6)
        g0, g1 = np.meshgrid(
            np.linspace(X[:, 0].min(), X[:, 0].max(), 30),
            np.linspace(X[:, 1].min(), X[:, 1].max(), 30),
        )
        zz = reg.predict(np.column_stack([g0.ravel(), g1.ravel()])).reshape(g0.shape)
        ax.plot_surface(g0, g1, zz, color="red", alpha=0.3)
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.set_zlabel(target)

    elif n == 3:
        fig = plt.figure(figsize=(13, 5))

        # Left: the 3 features in space, color = dependent variable
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="viridis")
        fig.colorbar(sc, ax=ax, label=target, shrink=0.6, pad=0.1)
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.set_zlabel(names[2])

        # Right: predicted vs actual (points on the dashed line = perfect fit)
        ax2 = fig.add_subplot(1, 2, 2)
        pred = reg.predict(X)
        ax2.scatter(y, pred, alpha=0.6)
        lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
        ax2.plot(lims, lims, "r--")
        ax2.set_xlabel(f"actual {target}")
        ax2.set_ylabel(f"predicted {target}")

    else:
        raise ValueError(f"Expected 1-3 features, got {n}")

    plt.tight_layout()
    plt.show()

def plot_residuals(reg, X, y, names=None, target="y"):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[1]
    names = names or [f"x{i + 1}" for i in range(n)]

    pred = reg.predict(X)
    resid = y - pred

    feature_cols = n if n > 1 else 0
    ncols = 1 + feature_cols + 1
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=True)

    # Residuals vs predicted
    axes[0].scatter(pred, resid, alpha=0.6)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel(f"predicted {target}")
    axes[0].set_ylabel("residual (actual - predicted)")

    # Residuals vs each feature
    for i in range(feature_cols):
        ax = axes[i + 1]
        ax.scatter(X[:, i], resid, alpha=0.6)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel(names[i])

    # Distribution of residuals
    axes[-1].hist(resid, bins=20, orientation="horizontal", edgecolor="black")
    axes[-1].axhline(0, color="red", linestyle="--")
    axes[-1].set_xlabel("count")

    fig.suptitle("Residuals")
    plt.tight_layout()
    plt.show()