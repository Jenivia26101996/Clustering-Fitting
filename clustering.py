# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 01:29:26 2023

@author: Huawei
"""
import pandas as pd
import wbgapi as wb
import sklearn
import seaborn as sns
from sklearn.datasets import make_blobs
from numpy import array, exp
import itertools as iter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sklearn.cluster as cluster
import sklearn.metrics as skmet

def worldbank(filename):
    df=pd.read_csv(filename,skiprows=(4))
    df_transpose=df.transpose()
    return df,df_transpose
a,b=worldbank("C:/Users/Huawei/Desktop/ADSAssign2/Clustering-Fitting/API_19_DS2_en_csv_v2_4773766.csv")
print(a)
print(b)

ecn_indc = ['NE.DAB.TOTL.ZS','NY.GDP.MKTP.CD']
cod_contry = ["BMU","CHE",'DNK','BGR','BGD','ARG','GBR','IND','BRA','JAM']
cli_indc=['EG.ELC.RNWX.KH','EN.ATM.CO2E.GF.KT']
ecn_data  = wb.data.DataFrame(ecn_indc, cod_contry, mrv=7)
cli_data  = wb.data.DataFrame(cli_indc, cod_contry, mrv=7)
#NE.DAB.TOTL.ZS: Total expenditure
#NY.GDP.MKTP.CD: USD GDP of a country
#EG.ELC.RNWX.KH:	Electricity production from renewable sources %
#EN.ATM.CO2E.GF.KT: Emissions of Carbon dioxide from fuel 
# ECNMY INDICATOR
ecn_data.columns = [b.replace('YR','') for b in ecn_data.columns]      
ecn_data=ecn_data.stack().unstack(level=1)                             
ecn_data.index.names = ['Ctry_Code', 'Year']                           
ecn_data.columns                                                     
ecn_data.fillna(0)
print(ecn_data.head(9))

# CLMATE INDICATOR
cli_data.columns = [c.replace('YR','') for c in cli_data.columns]      
cli_data=cli_data.stack().unstack(level=1)                             
cli_data.index.names = ['Ctry_Code', 'Year']                           
cli_data.columns                                                     
cli_data.fillna(0)
cli_data.head(9)

#Preprtion of the data
dfrm1=ecn_data.reset_index()
dfrm3=dfrm1.fillna(0)
dfrm2=cli_data.reset_index()
dfrm4=dfrm2.fillna(0)

#Getting the indicators for all the countries
dfrm = pd.merge(dfrm3, dfrm4)
X=dfrm.head(10)

print(X)

#Normalization of the dfrm values
df1 = dfrm.iloc[:,2:]
dfrm.iloc[:,2:] = (df1-df1.min())/ (df1.max() - df1.min())
dfrm.head(7)

#Clustering the value of total expenditure for different countries
plt.scatter(data=dfrm, x="Ctry_Code", y="NE.DAB.TOTL.ZS",cmap="cool")
plt.legend(loc='lower right')
plt.show()