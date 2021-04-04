################################################################################################
####################################### IMPORT LIBRARIES #######################################
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
# To Scale our data
from sklearn.preprocessing import scale

# To perform KMeans clustering 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# To perform Hierarchical clustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

####################################### IMPORT INPUT DATA #######################################
df2 = pd.read_json('C:/Users/aikaterini.biazi/Downloads/bq_results3.json', orient='split')
df2["order_time"] = pd.to_datetime(df2["submit_dt"])
(df2.dtypes)

############################# Perform some initial sanity checks ################################
### Count the unique values for each columns
df2.cuisine_parent.nunique()
# 9

def unique_counts(df2):
   for i in df2.columns:
       count = df2[i].nunique()
       print(i, ": ", count)
unique_counts(df2)

# order_id :  400000
# brand :  1
# submit_dt :  353609
# user_id :  162954
# shop_id :  3349
# city :  91
# cuisine_parent :  9
# basket :  2140
# order_time :  353609

### Check for nulls
df2.isnull().sum(axis=0)

# order_id          0
# brand             0
# submit_dt         0
# user_id           0
# shop_id           0
# city              0
# cuisine_parent    0
# basket            0
# order_time        0

print(df2.basket.min()) #0.0
print(len(df2[(df2['basket']==0)])) #9 NO NEED to perform a PCA analysis
check_basket = df2[(df2['basket']>0)]
print(check_basket.basket.min())  #0.2
print(df2.basket.max())  #151.85
print(df2['order_time'].min())  #2021-01-01 00:11:12+00:00
print(df2['order_time'].max())  #2021-01-31 23:59:58+00:00
#check the key
df2[["user_id","order_time"]].drop_duplicates() #400,000 distinct rows


### Check for correlation for all users among cuisine categories
cuisine_order_cnt = df2.groupby(['cuisine_parent',"user_id"],as_index=False).agg({"order_id":"count"}).sort_values('order_id', ascending=False) #"order_id":"count","basket":"sum"
cuisine_order_cnt = cuisine_order_cnt.pivot_table(index=["user_id"], 
                          columns='cuisine_parent', values = ['order_id'], aggfunc = np.sum, fill_value=0) 

cuisine_order_cnt.corr()
#cust_prod = pd.crosstab(df2['user_id'], df2['cuisine_parent']) #same result dif way


####################################### RFM Customer Segmentation #######################################

### RECENCY
# Since Recency is calculated from the last purchase till now and cause we have only data fro Jan i will calculate the max and avg duration that each customer did to make another payment 
df2["order_time_lag"] = df2.sort_values(by=['user_id','order_time']).groupby("user_id")['order_time'].shift(-1)

# if there is only 1 observation i will take the dif from end of month
cnt_rows_user = df2.groupby(["user_id"],as_index=False)\
                .agg( cnt_rows_user=('order_id', len))
df2 = pd.merge(df2, cnt_rows_user, how='left',on='user_id') 
df2.loc[df2['cnt_rows_user'] ==1, 'order_time_lag'] = pd.to_datetime("2021-01-31T23:59:59.000Z")
df2["order_freq"] =   df2["order_time_lag"]-df2["order_time"] 
# Check
df2[(df2.user_id == 99103596)].sort_values(by=['user_id','order_time']) 
# Check
df2[(df2.user_id == 39238)].sort_values(by=['user_id','order_time']) 
df2[(df2.user_id == 27462)].sort_values(by=['user_id','order_time'])


# CREATE RMF KPIS
rfm = df2.groupby(["user_id"],as_index=False)\
                .agg(max_freq = ('order_freq',max), order_cnt=('order_id', len),order_value=('basket', sum))

# rfm['InvoiceDate'] = rfm['InvoiceDate'].astype(int)
rfm.rename(columns={'max_freq': 'recency', 
                         'order_cnt': 'frequency', 
                         'order_value': 'monetary_value'}, inplace=True)

rfm['recency'] =rfm['recency'].dt.days
                         
                         
quantiles = rfm.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()     


def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1                    
                         
# If i had more custom functions, Functions should be included in different python file and this file should call it                 
                         
rfm['r_quartile'] = rfm['recency'].apply(RScore, args=('recency',quantiles,))
rfm['f_quartile'] = rfm['frequency'].apply(FMScore, args=('frequency',quantiles,))
rfm['m_quartile'] = rfm['monetary_value'].apply(FMScore, args=('monetary_value',quantiles,))
rfm.head()                       
                         
                         
rfm['RFMScore'] = rfm.r_quartile.map(str) + rfm.f_quartile.map(str) + rfm.m_quartile.map(str)
rfm.head()


rfm[rfm['RFMScore']=='111'].sort_values('monetary_value', ascending=False).head(10)
rfm[rfm['RFMScore']=='111'].sort_values('monetary_value', ascending=False).shape #5063


#SEGMENTS ACCORDING TO RFM 
rfm['segment'] = np.nan
rfm.loc[rfm['m_quartile'] ==1, 'segment'] = 'Highest Paying customer'
rfm.loc[(rfm['r_quartile'] ==1) & (rfm['f_quartile'] ==1) & (rfm['m_quartile'] ==1), 'segment'] = 'Highly engaged customer'
rfm.loc[(rfm['r_quartile'] !=1) & (rfm['f_quartile'] ==1) & (rfm['m_quartile'] !=1) , 'segment'] = 'Loyal customer'
rfm.loc[((rfm['f_quartile'] ==1) & (rfm['m_quartile'] ==3)) | ((rfm['f_quartile'] ==1) & (rfm['m_quartile'] ==4)), 'segment'] = 'Promising customer'
rfm.loc[(rfm['r_quartile'] ==1) & (rfm['f_quartile'] ==4)  , 'segment'] = 'New customer'
rfm.loc[(rfm['r_quartile'] ==4) & (rfm['f_quartile'] ==4)  , 'segment'] = 'Prior loyal customers'

####################################### RFM Customer Segmentation - SECOND METHODOLOGY (K-Means)#######################################
#SEGMENTS ACCORDING TO k-means 
                        
# Q1 = rfm.monetary_value.quantile(0.25)
# Q3 = rfm.monetary_value.quantile(0.75)
# IQR = Q3 - Q1
# rfm = rfm[(rfm.monetary_value >= (Q1 - 1.5*IQR)) & (rfm.monetary_value <= (Q3 + 1.5*IQR))]

# STANDARDIZATION
rfmnorm = rfm.drop(["user_id","r_quartile","f_quartile","m_quartile","RFMScore","segment"], axis=1)

from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
rfmnorm = standard_scaler.fit_transform(rfmnorm)


#KMeans starts with 2 clusters adding +1
from sklearn.cluster import KMeans
model_clus5 = KMeans(n_clusters = 4, max_iter=50)
model_clus5.fit(rfmnorm)

from sklearn.metrics import silhouette_score
sse_ = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k).fit(rfmnorm)
    sse_.append([k, silhouette_score(rfmnorm, kmeans.labels_)])

plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1]) # check how similar are the clusters to one each other 

# SSD -> sum of squared distances
ssd = []
for num_clusters in list(range(1,10)):
    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)
    model_clus.fit(rfmnorm)
    ssd.append(model_clus.inertia_)

plt.plot(ssd)# i would choose 5 clusters

#MY MODEL
model_clus5 = KMeans(n_clusters = 3, max_iter=50)
model_clus5.fit(rfmnorm)

#MAPPING TO THE DATASET
# rfmnorm2 = rfm.drop(["r_quartile","f_quartile","m_quartile","RFMScore","segment"], axis=1)
rfmnorm_KM = pd.concat([rfm, pd.Series(model_clus5.labels_)], axis=1)
rfmnorm_KM.columns = ['user_id', 'recency', 'frequency', 'monetary_value',"r_quartile","f_quartile","m_quartile","RFMScore","segment", 'ClusterID']

km_clusters_recency = 	pd.DataFrame(rfmnorm_KM.groupby(["ClusterID"]).recency.mean())
km_clusters_frequency = 	pd.DataFrame(rfmnorm_KM.groupby(["ClusterID"]).frequency.mean())
km_clusters_amount = 	pd.DataFrame(rfmnorm_KM.groupby(["ClusterID"]).monetary_value.mean())
km_clusters_total_amount = 	pd.DataFrame(rfmnorm_KM.groupby(["ClusterID"]).monetary_value.sum())


#REPORTING
df = pd.concat([pd.Series([0,1,2,3,4,5,6]), km_clusters_recency,  km_clusters_frequency, km_clusters_amount,km_clusters_total_amount ], axis=1)
df.columns = ["ClusterID", "Recency_mean", "Frequency_mean","Amount_mean", "Total_Ammount"]
df.info()

sns.barplot(x=df.ClusterID, y=df.Amount_mean)
sns.barplot(x=df.ClusterID, y=df.Frequency_mean)
sns.barplot(x=df.ClusterID, y=df.Recency_mean)

##KEEP IT FOR BPI
df_by_cuisine = df_orig.groupby(["user_id","cuisine_parent"],as_index=False)\
                .agg(max_freq = ('order_freq',max), order_cnt=('order_id', len),order_value=('basket', sum))
df_by_cuisine['max_freq'] =df_by_cuisine['max_freq'].dt.days


#SAVE
save_path = 'C:/Users/aikaterini.biazi/Downloads'

current_time = datetime.datetime.now() 
suffix = str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day) + "_" + str(current_time.hour) +"."+ str(current_time.minute) 

rfmnorm_KM.to_csv(save_path + "/" + 'RMF_analysis_'+suffix+'.csv', index = None)
df_by_cuisine.to_csv(save_path + "/" + 'extra_info_pbi_'+suffix+'.csv', index = None)