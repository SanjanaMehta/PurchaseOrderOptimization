#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r"C:\Users\Sanjana\Downloads\my documents\projects\surgical supply\final.csv")


# In[3]:


df.info()


# In[4]:


# Drop duplicate columns
df = df.T.drop_duplicates().T


# In[5]:


dropcols= ['Item_Description.1','Department.1']
df=df.drop(columns=dropcols)


# In[6]:


df['PO_Date']=pd.to_datetime(df['PO_Date'])


# In[7]:


df['Delivery_Date']=pd.to_datetime(df['Delivery_Date'])


# In[8]:


df['Expected_Delivery_Date']=pd.to_datetime(df['Expected_Delivery_Date'])


# In[9]:


df.info()


# In[10]:


df['PO_Status'].value_counts()


# In[11]:


df['Delay_Days']=df['Delivery_Date']-df['Expected_Delivery_Date']
df['Delay_Days'] = df['Delay_Days'].dt.days


# In[12]:


df['Diff_Quantity']=df['Quantity_Ordered']-df['Quantity_Received']
df.head()


# In[13]:


df['Department'].value_counts()


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,8))
sns.barplot(data=df,x='Item_Description',y='Delay_Days',color='blue')


# In[15]:


plt.figure(figsize=(12,8))
sns.histplot(df['Diff_Quantity'])


# In[16]:


sns.relplot(data=df, x='Delay_Days',y='Late_Shipments')


# In[17]:


df['Request_Initiated_Date']=pd.to_datetime(df['Request_Initiated_Date'])


# In[18]:


df['Approval_Date']=pd.to_datetime(df['Approval_Date'])


# In[19]:


df['Dispatch_Date']=pd.to_datetime(df['Dispatch_Date'])


# In[20]:


df['Acknowledgment_Date']=pd.to_datetime(df['Acknowledgment_Date'])


# In[21]:


df['Goods_Received_Date']=pd.to_datetime(df['Goods_Received_Date'])


# In[22]:


df['Delay_Reason'].value_counts()


# In[23]:


plt.figure(figsize=(12,8))

sns.histplot(df['Delay_Reason'])


# In[24]:


sns.histplot(df['PO_Status'])


# In[25]:


df['Vendor_Rating']=df['Vendor_Rating'].replace({'A':'1','B':'2','C':'3'})
df.head()


# In[26]:


plt.figure(figsize=(20,5))
plt.subplot(1,4,1)
sns.barplot(data=df,x="Vendor_ID",y="Avg_PO_Cycle_Time")
plt.subplot(1,4,2)
sns.barplot(data=df,x="Vendor_ID",y="Late_Shipments")
plt.subplot(1,4,3)
sns.barplot(data=df,x="Vendor_ID",y="OTIF_Score (%)")
plt.subplot(1,4,4)
sns.barplot(data=df,x="Vendor_ID",y="Fill_Rate (%)")


# In[27]:


plt.figure(figsize=(20,5))
plt.pie(df['Vendor_Rating'],labels=df['Vendor_Name'])


# In[28]:


df['PO_Status']=df['PO_Status'].replace({'Closed':'0','Open':'1','Delayed':'2','Cancelled':'3'})


# In[29]:


df['Stockout_Flag'].value_counts()


# In[30]:


df['Quantity_Ordered']=df['Quantity_Ordered'].astype(int)
df['Quantity_Received']=df['Quantity_Received'].astype(int)
df['PO_Status']=df['PO_Status'].astype(int)
df['OTIF_Score (%)']=df['OTIF_Score (%)'].astype(int)
df['Late_Shipments']=df['Late_Shipments'].astype(int)
df['Fill_Rate (%)']=df['Fill_Rate (%)'].astype(int)
df['Avg_PO_Cycle_Time']=df['Avg_PO_Cycle_Time'].astype(int)
df['Vendor_Rating']=df['Vendor_Rating'].astype(int)
df['Current_Stock_Level']=df['Current_Stock_Level'].astype(int)
df['Reorder_Level']=df['Reorder_Level'].astype(int)
df['Forecasted_Demand']=df['Forecasted_Demand'].astype(int)
df['Stockout_Flag']=df['Stockout_Flag'].astype(int)
df['Delay_Days']=df['Delay_Days'].astype(int)
df['Diff_Quantity']=df['Diff_Quantity'].astype(int)


# In[31]:


plt.figure(figsize=(20,20))
corr_mat=df[['Quantity_Ordered','Quantity_Received','PO_Status','OTIF_Score (%)','Late_Shipments','Fill_Rate (%)','Avg_PO_Cycle_Time','Vendor_Rating','Current_Stock_Level','Reorder_Level','Forecasted_Demand','Stockout_Flag','Delay_Days','Diff_Quantity']].corr()
sns.heatmap(corr_mat,annot=True,cmap='viridis')


# In[32]:


df.info()


# In[33]:


df['Last_Reorder_Date']=pd.to_datetime(df['Last_Reorder_Date'])


# In[34]:


df.to_csv("Final.csv")


# In[35]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score


# In[36]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[37]:


s=(df.dtypes=='object')
objcols=list(s[s].index)


# In[38]:


df1=df.copy()
LE=LabelEncoder()
for i in objcols:
    df1[i]=df[[i]].apply(LE.fit_transform)


# In[39]:


t=(df.dtypes=='datetime64[ns]')
dtcols=list(t[t].index)
for i in dtcols:
    df1[i]=df[[i]].apply(LE.fit_transform)


# In[40]:


df1.info()


# In[41]:


df['Stockout_Flag'].value_counts()


# In[53]:


df2=df[['Quantity_Ordered','Quantity_Received','PO_Status','OTIF_Score (%)','Late_Shipments','Fill_Rate (%)','Avg_PO_Cycle_Time','Vendor_Rating','Current_Stock_Level','Reorder_Level','Forecasted_Demand','Stockout_Flag','Delay_Days','Diff_Quantity']]


# In[54]:


X=df2.drop('Stockout_Flag',axis=1)
y=df2['Stockout_Flag']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.24,random_state=42)


# In[55]:


model = RandomForestRegressor(n_estimators=100, random_state=42)


# In[56]:


model.fit(X_train, y_train)


# In[57]:


y_pred = model.predict(X_test)


# In[58]:


y_pred

