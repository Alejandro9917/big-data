from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as pt
import pathlib


def plotCluster2D(x, y, ax):
  y_uniques = pd.Series(y).unique()
  
  for i in y_uniques:
    x[y==i].plot(
      title = f"{len(y_uniques)} Clusters",
      kind = "scatter",
      x = "x1",
      y = "x2", 
      marker = f"${i}$",
      ax = ax
    )


base_path = str(pathlib.Path(__file__).parent.resolve()) + "/ecomerce.csv"
ecommerce_data = pd.read_csv(base_path)

# Clean data
ecommerce_data.dropna(inplace=True)
ecommerce_data = ecommerce_data[["price", "review_score", "review_count"]]


# Reduccion de dimensionalidad en variables numericas
df_numerico = ecommerce_data.select_dtypes(include="number")


#Reduccion de dimensionalidad
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_numerico)


#Conver principals components to Dataframe
pca_df = pd.DataFrame(pca_components, columns = ["PC1", "PC2"])
print(pca_df)


#Clusterizacion
kmodel = KMeans(n_clusters=4, random_state=20)
y_pred = kmodel.fit_predict(pca_df)
pca_df["Clusters"] = y_pred
fig, ax = pt.subplots(1,1,figSize=(7,5))
x, y = pca_df[["PC1", "PC2"]], pca_df["Clusters"]
plotCluster2D(x,y, ax)