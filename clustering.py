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
from scipy.optimize import curve_fit
import scipy.optimize as opt
import err_ranges as err




hiv_data=pd.read_csv("C:/Users/Huawei/Desktop/ADSAssign2/Clustering-Fitting/AIDS_data.csv",skiprows=(4))
deathrate_data=pd.read_csv("C:/Users/Huawei/Desktop/ADSAssign2/Clustering-Fitting/deathrate_data.csv",skiprows=(4))
tb_data=pd.read_csv("C:/Users/Huawei/Desktop/ADSAssign2/Clustering-Fitting/tuberculosis_data.csv",skiprows=(4))
under_data=pd.read_csv("C:/Users/Huawei/Desktop/ADSAssign2/Clustering-Fitting/undernourishment_data.csv",skiprows=(4))
hiv_data.rename(columns = {'2015' : 'Aids Prevalance data in 2015'}, inplace = True) #Rename the column Name
deathrate_data.rename(columns = {'2015' : 'Death rate data in 2015'}, inplace = True) #Rename the column Name
tb_data.rename(columns = {'2015' : 'Tuberculosis Prevalance data in 2015'}, inplace = True) #Rename the column Name
under_data.rename(columns = {'2015' : 'Undernourishment Prevalance data in 2015'}, inplace = True) #Rename the column Name
merge_df = pd.merge(hiv_data, hiv_data, on=['Country Name'])
merge_df = pd.merge(hiv_data, deathrate_data, on=['Country Name'])
merge_df = pd.merge(merge_df, tb_data, on='Country Name')
merge_df = pd.merge(merge_df, under_data, on='Country Name')
merge_df.to_csv('merge_df.csv', index=False)

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
def heatmap(data):
    sns.heatmap(data,cmap = 'Wistia', annot = True)
    return
plt.figure(figsize = (10,8))
heatmap(a.corr())
plt.title('Heatmap for the Data')
plt.savefig("heatmap.png",bbox_inches="tight")
plt.show()
    
def matrixplot(df):
    sm=pd.plotting.scatter_matrix(df,figsize=(11,11),diagonal='kde',alpha=0.2)
    #Change label rotation
    plt.figure(figsize = (10,8),dpi=144)
    plt.tight_layout() # helps to avoid overlap of labels
    [s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
    [s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]

#May need to offset label when rotating to prevent overlap of figure
    [s.get_yaxis().set_label_coords(-0.7,0.5) for s in sm.reshape(-1)]
#Hide all ticks
    [s.set_xticks(()) for s in sm.reshape(-1)]
    [s.set_yticks(()) for s in sm.reshape(-1)]
    plt.savefig("matrix.png",bbox_inches="tight")
    plt.show()
    return
matrixplot(a)

"""Clustering"""
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
plt.savefig("cluster.png")
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
# #FITTING

def read_data(file_name, country, save_file):  # Defind a fuction to read the files
    df = pd.read_csv(file_name, header = [2]) # To read the file
    #df = df[df['Indicator Name'].isin([indicator])].reset_index(drop = True)  # To filter the data with required indicator
    df = df[df["Country Name"].isin([country])].reset_index(drop = True)  # To filter the data with required country
    df = df.T.reset_index(drop = False)  # To transpose the data
    new_col1 = 'Year' 
    new_col2 = df.iloc[2,1]
    df.columns = [new_col1, new_col2]  # Set the columns name
    df = df.iloc[4:,:].reset_index(drop = True)  # To remove unnecessary rows
    df = df.dropna().reset_index(drop = True)  # To drop NaN values
    df = df.astype(float)  # To convert the data into float
    print(df)  # To print the data
    df.to_excel(save_file)  # To save the file
    return df

df1 = read_data('C:/Users/Huawei/Desktop/ADSAssign2/Clustering-Fitting/tuberculosis_data.csv', 'United Kingdom', 'df1.xlsx')  # To call the fuction & store in df1
df2 = read_data('C:/Users/Huawei/Desktop/ADSAssign2/Clustering-Fitting/deathrate_data.csv', 'United Kingdom', 'df2.xlsx')  # To call the fuction & store in df2
df3 = read_data('C:/Users/Huawei/Desktop/ADSAssign2/Clustering-Fitting/tuberculosis_data.csv', 'United States', 'df3.xlsx')  # To call the fuction & store in df3
df4 = read_data('C:/Users/Huawei/Desktop/ADSAssign2/Clustering-Fitting/deathrate_data.csv', 'United States', 'df4.xlsx')  # To call the fuction & store in df4

def exp_growth(t, scale, growth):
    # Computing the  exponential function with scale and growth
    f = scale * np.exp(growth * (t-1950))
    return f
def logistics(t, scale, growth, t0):
    #Computing the logistics function with scale, growth rate and time of the turning point as free parameters
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    return f


df_tb = pd.read_excel("df3.xlsx") # Reads the file with population data into dataframe
print(df_tb) # printing the df_imports

# fitting the exponential growth
imports, covar = opt.curve_fit(exp_growth, df_tb["Year"],df_tb["Incidence of tuberculosis (per 100,000 people)"])

# Plotting the first attempt
print("Fit parameter", imports)
df_tb["imp_exp"] = exp_growth(df_tb["Year"], *imports)
plt.figure()
plt.plot(df_tb["Year"], df_tb["Incidence of tuberculosis (per 100,000 people)"], label="data")
plt.plot(df_tb["Year"], df_tb["imp_exp"], label="fit")
plt.legend()
plt.title("First fit attempt")
plt.xlabel("year")
plt.ylabel("Incidence of tuberculosis (per 100,000 people)")
plt.show()
print()

# Finding a start approximation
popt = [4e8, 0.01]
df_tb["pop_exp"] = exp_growth(df_tb["Year"], *popt)
plt.figure()
plt.plot(df_tb["Year"], df_tb["Incidence of tuberculosis (per 100,000 people)"], label="data")
plt.plot(df_tb["Year"], df_tb["pop_exp"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("Imports of goods and services (% of GDP)")
plt.title("Improved start value")
plt.show()

# fit exponential growth
imports, covar = opt.curve_fit(exp_growth, df_tb["Year"],
df_tb["Incidence of tuberculosis (per 100,000 people)"], p0=[4e8, 0.02])

print("Fit parameter", imports)
df_tb["imp_exp"] = exp_growth(df_tb["Year"], *imports)
plt.figure()
plt.plot(df_tb["Year"], df_tb["Incidence of tuberculosis (per 100,000 people)"], label="data")
plt.plot(df_tb["Year"], df_tb["imp_exp"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("Incidence of tuberculosis (per 100,000 people)")
plt.title("Final fit exponential growth")
plt.savefig("expfit.png",bbox_inches="tight")
plt.show()
print()

# Increase scale factor and growth rate until rough fit
imports, covar = opt.curve_fit(logistics, df_tb["Year"], df_tb["Incidence of tuberculosis (per 100,000 people)"],
p0=(2e9, 0.05, 1990.0))
print("Fit parameter", imports)
df_tb["imp_log"] = logistics(df_tb["Year"], *imports)
plt.figure()
plt.title("logistics function")
plt.plot(df_tb["Year"], df_tb["Incidence of tuberculosis (per 100,000 people)"], label="data")
plt.plot(df_tb["Year"], df_tb["imp_log"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("Incidence of tuberculosis (per 100,000 people)")
plt.savefig("logisticsfit.png",bbox_inches="tight")
plt.show()

# Function for returning upper and lower limits of the error ranges.
sigma = np.sqrt(np.diag(covar))
print(sigma)
low, up = err.err_ranges(df_tb["Year"], logistics, imports, sigma)
plt.figure()
plt.title("Incidence of tuberculosis (per 100,000 people)")
plt.plot(df_tb["Year"], df_tb["Incidence of tuberculosis (per 100,000 people)"], label="data")
plt.plot(df_tb["Year"], df_tb["imp_log"], label="fit")
plt.fill_between(df_tb["Year"], low, up, alpha=0.7)
plt.legend()
plt.xlabel("Year")
plt.ylabel("Incidence of tuberculosis (per 100,000 people)")
plt.show()

# Forcasting the future values
print("Forcasted IIncidence of tuberculosis (per 100,000 people)")
low, up = err.err_ranges(2030, logistics, imports, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", mean, "+/-", pm)
low, up = err.err_ranges(2040, logistics, imports, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", mean, "+/-", pm)
low, up = err.err_ranges(2050, logistics, imports, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2050:", mean, "+/-", pm)

    

