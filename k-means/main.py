from sklearn.datasets import make_blobs
import pandas as pd
import matplotlib.pyplot as pt

x, y = make_blobs(
    n_samples=100, centers=4, n_features=2, cluster_std=[1, 1.5, 2, 2], random_state=7
)

print(x)

df_blobs = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y})


def plotClusters2D(x, y, ax):
    y_uniques = pd.Series(y).unique()

    for i in y_uniques:
        x[y == i].plot(
            title=f"{len(y_uniques)} Clusters",
            kind="scatter",
            x="x1",
            y="x2",
            marker=f"${i}$",
            ax=ax,
        )


fig, ax = pt.subplots(1, 1, figsize=(7, 5))
x, y = df_blobs[["x1", "x2"]], df_blobs["y"]

plotClusters2D(x, y, ax)
