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
Created a function data to manipulate the data using pandas dataframes which takes a csv file as argument, 
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
hiv_data=pd.read_csv("C:/Users/Huawei/Desktop/ADSAssign2/Clustering-Fitting/AIDS_data.csv",skiprows=(4))
deathrate_data=pd.read_csv("C:/Users/Huawei/Desktop/ADSAssign2/Clustering-Fitting/deathrate_data.csv",skiprows=(4))
tb_data=pd.read_csv("C:/Users/Huawei/Desktop/ADSAssign2/Clustering-Fitting/tuberculosis_data.csv",skiprows=(4))
under_data=pd.read_csv("C:/Users/Huawei/Desktop/ADSAssign2/Clustering-Fitting/undernourishment_data.csv",skiprows=(4))

# def merge():
#     a=hiv_data.rename(columns = {'2015' : 'Aids Prevalance data in 2015'}, inplace = True) #Rename the column Name
#     b=deathrate_data.rename(columns = {'2015' : 'Death rate data in 2015'}, inplace = True) #Rename the column Name
#     c=tb_data.rename(columns = {'2015' : 'Tuberculosis Prevalance data in 2015'}, inplace = True) #Rename the column Name
#     d=under_data.rename(columns = {'2015' : 'Undernourishment Prevalance data in 2015'}, inplace = True) #Rename the column Name
#     return 

# merge_df = pd.merge(hiv_data, hiv_data, on=['Country Name'])
# merge_df = pd.merge(hiv_data, deathrate_data, on=['Country Name'])
# merge_df = pd.merge(merge_df, tb_data, on='Country Name')
# merge_df = pd.merge(merge_df, under_data, on='Country Name')

# merge_df.to_csv('merge_df.csv', index=False)
# print(merge_df)
def data(filename):
    df=pd.read_csv(filename)
    df_country=df['Country Name']
    df=df.iloc[:,[59,125,191,257]]
    df = df.fillna(0)
    print(df)
    df.insert(loc=0,column='Country Name',value=df_country)
    df=df.dropna(axis=1)
    df_t=df.set_index('Country Name').T
    #print(df)
    return df,df_t 
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
def heatmap():
    plt.figure(figsize = (10,8))
    sns.heatmap(a.corr(), cmap = 'Wistia', annot = True)
    plt.title('Heatmap for the Data')
    plt.savefig("heatmap.png",bbox_inches="tight")
    plt.show()
    return
def matrixplot():
    plt.figure(figsize = (10,8),dpi=144)
    pd.plotting.scatter_matrix(a, figsize=(9.0, 9.0))
    plt.xticks(rotation=45)
    plt.tight_layout() # helps to avoid overlap of labels
    plt.savefig("matrix.png",bbox_inches="tight")
    plt.show()
    return



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
n_clusters=4
kmeans = cluster.KMeans(n_clusters=4)
kmeans.fit(df_fit)
# extract labels and cluster centres
labels = kmeans.labels_
#Extract & Store the group of country according to labels
country_group0 = a[labels == 0]
country_group1 = a[labels == 1]
country_group0.to_excel('country_group0.xlsx')
country_group1.to_excel('country_group1.xlsx')
cen = kmeans.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
# Individual colours can be assigned to symbols. The label l is used to the␣
# l-th number from the colour table.
scatter=plt.scatter(df_fit["Tuberculosis Prevalance data in 2015"], df_fit["Death rate data in 2015"], c=labels, cmap="Accent",s = 50 )
# colour map Accent selected to increase contrast between colours
# show cluster centres
for ic in range(4):
    xc, yc = cen[ic,:]
plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("Tuberculosis Prevalance data in 2015")
plt.ylabel("Death rate data in 2015")
plt.title("4 clusters")
plt.colorbar(scatter, label="Cluster label")
plt.legend(handles = scatter.legend_elements()[0], labels = ["Cluster " + str(i + 1) for i in range(n_clusters)])
plt.show()
# #-----------------------
# # Plot for three clusters
n_clusters=3
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(df_fit)
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
# Individual colours can be assigned to symbols. The label l is used to the␣
# l-th number from the colour table.
scatter=plt.scatter(df_fit["Tuberculosis Prevalance data in 2015"], df_fit["Death rate data in 2015"], c=labels, cmap="Accent")
# colour map Accent selected to increase contrast between colours
# show cluster centres
for ic in range(3):
    xc, yc = cen[ic,:]
plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("Tuberculosis Prevalance data in 2015")
plt.ylabel("Death rate data in 2015")
plt.title("3 clusters")
plt.colorbar(scatter, label="Cluster label")
plt.legend(handles = scatter.legend_elements()[0], labels = ["Cluster " + str(i + 1) for i in range(n_clusters)])
plt.show()

