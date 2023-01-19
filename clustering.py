import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sns
from sklearn import cluster
from sklearn import preprocessing
import plotly.express as px
import sklearn.metrics as skmet
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
"""
Created a function electric to manipulate the data using pandas dataframes which takes a csv file as argument, 
reads a dataframe in Worldbank format which is electric consumption and returns two dataframes: 
one with years as columns and one with countries as columns.
iloc:for selective columns in dataframe
fillna(0):will replace Nan values with 0
Transposed(T): is used to return transposed columns
kind='line': line plot function in pandas
bbox_to_anchor: to place leegend outside the box
label:display labels
savefig:saves image in the directory
linestyle: we can display different types of line
"""

def data(filename):
    df_electric=pd.read_csv(filename)
    df_country=df_electric['Country Name']
    df_electric=df_electric.iloc[:,[59,125,191,257]]
    df_electric = df_electric.fillna(0)
    print(df_electric)
    df_electric.insert(loc=0,column='Country Name',value=df_country)
    df_electric=df_electric.dropna(axis=1)
    df_electrict=df_electric.set_index('Country Name').T
    #print(df)
    return df_electric,df_electrict 
a,b=data("merge_df.csv")
print(a)
print(b)
def norm(array):
    """Returns array normalised to [0,1]. Array can be a numpy array or a column of a dataframe"""
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    return scaled
  
def norm_df(df, first=0, last=None):
    """Returns all columns of the dataframe normalised to [0,1] with the
exception of the first (containing the names)
Calls function norm to do the normalisation of one column, but
doing all in one function is also fine.
First, last: columns from first to last (including) are normalised.
Defaulted to all. None is the empty entry. The default corresponds
"""
# iterate over all numerical columns
    for col in df.columns[first:last]: # excluding the first column
     df[col] = norm(df[col])
    return df


# heatmap
plt.figure(figsize = (10,8))
sns.heatmap(a.corr(), cmap = 'Wistia', annot = True)
plt.title('Heatmap for the Data')
plt.show()

plt.figure(figsize = (10,8),dpi=144)
pd.plotting.scatter_matrix(a, figsize=(9.0, 9.0))
plt.tight_layout() # helps to avoid overlap of labels
plt.show()



# extract columns for fitting
df_fit = a[['Tuberculosis Prevalance data in 2015','Death rate data in 2015']].copy()
# normalise dataframe and inspect result
# normalisation is done only on the extract columns. .copy() prevents
# changes in df_fit to affect df_fish. This make the plots with the
# original measurements
df_fit = norm_df(df_fit)
print(df_fit.describe())
print()

for ic in range(2, 7):
# set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)
# extract labels and calculate silhoutte score
labels = kmeans.labels_
print (ic, skmet.silhouette_score(df_fit, labels))
# Plot for four clusters
kmeans = cluster.KMeans(n_clusters=4)
kmeans.fit(df_fit)
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
# Individual colours can be assigned to symbols. The label l is used to the␣
# l-th number from the colour table.
plt.scatter(df_fit["Tuberculosis Prevalance data in 2015"], df_fit["Death rate data in 2015"], c=labels, cmap="Accent",s = 50 )
# colour map Accent selected to increase contrast between colours
# show cluster centres
for ic in range(4):
    xc, yc = cen[ic,:]
plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("Tuberculosis Prevalance data in 2015")
plt.ylabel("Death rate data in 2015")
plt.title("4 clusters")
plt.show()
# #-----------------------
# # Plot for five clusters
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(df_fit)
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
# Individual colours can be assigned to symbols. The label l is used to the␣
# l-th number from the colour table.
plt.scatter(df_fit["Tuberculosis Prevalance data in 2015"], df_fit["Death rate data in 2015"], c=labels, cmap="Accent")
# colour map Accent selected to increase contrast between colours
# show cluster centres
for ic in range(3):
    xc, yc = cen[ic,:]
plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("Tuberculosis Prevalance data in 2015")
plt.ylabel("Death rate data in 2015")
plt.title("3 clusters")
plt.show()