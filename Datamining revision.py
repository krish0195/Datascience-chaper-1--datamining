#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.cluster.hierarchy import dendrogram,linkage,fcluster

from sklearn.cluster import AgglomerativeClustering,KMeans

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import fcluster

from scipy.stats import zscore



# In[27]:


en=pd.read_csv(r"E:\data science\GREAT LAKES\back to studies\videaos\revision\datamining\week 1 clustering\Engg_College_Data.csv")


# In[28]:


en.head()


# In[29]:


en.drop("SR_NO",axis=1,inplace=True)


# In[30]:


en.shape


# In[31]:


en.info()


# In[32]:


en.Engg_College.nunique()


# In[33]:


en.describe()


# In[34]:


lk=linkage(en.drop("Engg_College",axis=1),method="ward")
dn=dendrogram(lk,truncate_mode="lastp",p=4)


# In[35]:


cluster=fcluster(lk,criterion="maxclust",t=3)


# In[36]:


en["divisive_cluster"]=cluster


# In[37]:


en


# In[38]:


en.groupby("cluster").mean().round()


# In[39]:


ag=AgglomerativeClustering( n_clusters=3)


# In[40]:


cl=ag.fit_predict(en.drop("Engg_College",axis=1))
en["aglo_cluster"]=cl


# In[41]:


en


# # en.groupby("cluster_aglo").mean().round()

# In[85]:


Bank data set


# In[86]:


bk=pd.read_csv(r"E:\data science\GREAT LAKES\back to studies\videaos\revision\datamining\week 1 clustering\bank.csv")


# In[87]:


bk.describe().round()


# In[88]:


bk.duplicated().sum()


# In[89]:


bks=bk.drop("Bank",axis=1).apply(zscore)
bks.describe().round()


# In[90]:


km=KMeans(n_clusters=6,random_state=0)


# In[91]:


cluster=km.fit_predict(bks)
km.inertia_


# In[92]:


bk["k_means_cluster"]=cluster
bk.head()


# In[96]:


bk.groupby("k_means_cluster").mean().round()


# In[93]:


ws=[]
for i in np.arange(1,25):
    km=KMeans(n_clusters=i,random_state=0)
    km.fit(bks)
    ws.append(km.inertia_) 


# In[80]:


ws


# In[81]:


plt.plot(ws)


# In[97]:


silhouette_score(bks,labels=km.labels_)


# In[ ]:




