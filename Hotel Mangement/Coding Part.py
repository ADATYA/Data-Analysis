#!/usr/bin/env python
# coding: utf-8

# Data Analysis Project Steps:
# 1. Create a Problem Statement.
# 2. Identify the data you want to analyze.
# 3. Explore and Clean the data.
# 4. Analyze the data to get useful insights.
# 5. Present the data in terms of reports or dashboards using visualization.
# 
# <b>Video link:https://youtu.be/obJZ1rB7TKc?si=6MRSIrmqZpR97Zn_<br>
# <b>Dataset link:https://drive.google.com/file/d/1-QwWigkene6K-EXRQTXEEEghq9SMiirb/view
#         

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Import Datasets

# In[2]:


df=pd.read_csv('hotel_bookings 2.csv')


# # EDA and Data Cleaning

# In[3]:


df.head(5)


# In[4]:


df.tail(5)


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.info()


# In this case there is totaly '119390' totla number of non null values.But here is some issue in 'reservation_status_date' are object but this is show as a object from ,now it is converting it into date time function.

# In[8]:


# changin datatime function object to date time function
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])


# In[9]:


df.info()


# In[10]:


# All Statistical Value are showing hear
df.describe()


# In[11]:


# Now I find spacific object value..
df.describe(include="object")


# In[12]:


# Here Is the object files are create spacific value of object
for col in df.describe(include='object').columns:
    print(col)
    print(df[col].unique())
    print("-"*50)


# In[13]:


df.isnull().sum()


# Here agent and company are over flow data in this dataset so I will drop this data using dropna

# In[14]:


df.drop(['agent','company'], axis= 1, inplace=True)
df.dropna(inplace=True)


# In[15]:


df.isnull().sum()


# In[16]:


df.describe()


# In[17]:


df=df[df['adr']<5000]


# In[18]:


df.describe()


# # Data Analysis and Visualizasion
# 

# In[19]:


#If we show the percentage value from there.
cancelled_per = df['is_canceled'].value_counts(normalize=True)
cancelled_per


# In[20]:


plt.figure(figsize=(5,4))
plt.title("Reservation Systrm")
plt.bar(["not_cancled","cancled"],df['is_canceled'].value_counts(),edgecolor ='r',color='b',width=0.7)
plt.show()


# <b>What is Hue in seaboen?</b><br>
# In Seaborn, hue is a parameter that adds a third dimension to the plot by coloring different subsets of data based on a categorical variable, allowing for visual differentiation between groups.

# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
ax1=sns.countplot(x='hotel', hue='is_canceled', data=df, palette='Blues')
legend_lebels,_ = ax1.get_legend_handles_labels()
ax1.legend(bbox_to_anchor=(1, 1))
plt.title("Reservation status is diffurent hotels", size=20)
plt.legend(['not_canceled','calceled'])
plt.xlabel('hotel')
plt.ylabel('number of reservation')
plt.show()


# In[22]:


#We are finding the percentage value from values.
#filterr the value for 'Resort Hotel'
resort_hotel = df[df['hotel'] == 'Resort Hotel']
resort_hotel['is_canceled'].value_counts(normalize = True)


# In[23]:


#filterr the value for 'City Hotel'
city_hotel = df[df['hotel'] == 'City Hotel']
city_hotel['is_canceled'].value_counts(normalize = True)


# In[24]:


#All resurvation date groupd in this column..
resort_hotel = resort_hotel.groupby("reservation_status_date")[['adr']].mean()
city_hotel = city_hotel.groupby("reservation_status_date")[['adr']].mean()


# In[25]:


plt.figure(figsize=(15,7))
plt.title('Avarage Daily Rate in City and Resort Hotels',fontsize=20)
plt.plot(resort_hotel.index,resort_hotel['adr'],label="Resort Hotel")
plt.plot(city_hotel.index,city_hotel['adr'],label="City Hotel")
plt.legend(fontsize=15)
plt.show()


# In[26]:


# Per month hotel cancellation status.
df['month'] = df['reservation_status_date'].dt.month  ## changing date resurvation to month
plt.figure(figsize=(10,5))
axt=sns.countplot(x='month',hue='is_canceled',data=df,palette='bright')
legend_lebels,_ = ax1.get_legend_handles_labels()
ax1.legend(bbox_to_anchor=(1, 1))
plt.title("Reservation status per month", size=20)
plt.legend(['not_canceled','calceled'])
plt.xlabel('month')
plt.ylabel('number of reservations')
plt.show()


# In[27]:


#Hypothisis for canceled for which month is very pic for canceld hotel 
plt.figure(figsize=(15,7))
plt.title("ADR per month",fontsize=20)
sns.barplot('month','adr', data=df[df['is_canceled'] == 1].groupby('month')[['adr']].sum().reset_index())
#plt.legend(fontsize=20)
plt.show()


# In[28]:


cancelled_data=df[df['is_canceled'] ==1]
top_10_country = cancelled_data['country'].value_counts()[:10]
plt.figure(figsize=(15,7))
plt.title('Top 10 country with reservation canceled')
plt.pie(top_10_country, autopct = '%.2f',labels = top_10_country.index)
plt.show()


# In[29]:


#Full Statement of Hotes
df['market_segment'].value_counts()


# In[30]:


#Percentage of all top hotels
df['market_segment'].value_counts(normalize = True)


# In[31]:


# Canceled hotel information segmentatinon with percentage.
cancelled_data['market_segment'].value_counts(normalize = True)


# In[ ]:





# In[32]:


# For canceled bookings
cancelled_df_adr = cancelled_data.groupby('reservation_status_date')[['adr']].mean()
cancelled_df_adr.reset_index(inplace=True)
cancelled_df_adr.sort_values('reservation_status_date', inplace=True)

# For non-canceled bookings
not_cancelled_data = df[df['is_canceled'] == 0]
not_cancelled_df_adr = not_cancelled_data.groupby("reservation_status_date")[['adr']].mean()
not_cancelled_df_adr.reset_index(inplace=True)
not_cancelled_df_adr.sort_values('reservation_status_date', inplace=True)

# Plotting
plt.figure(figsize=(15, 7))
plt.title("Average Daily Rate",fontsize=30)

# Plotting ADR for not canceled bookings
plt.plot(not_cancelled_df_adr["reservation_status_date"], not_cancelled_df_adr['adr'], label='not cancelled')

# Plotting ADR for canceled bookings
plt.plot(cancelled_df_adr['reservation_status_date'], cancelled_df_adr['adr'], label='cancelled')

plt.legend(fontsize=20)
plt.show()

#ref.chatgpt


# In[33]:


# Ensure that reservation_status_date is in datetime format
cancelled_df_adr['reservation_status_date'] = pd.to_datetime(cancelled_df_adr['reservation_status_date'])
not_cancelled_df_adr['reservation_status_date'] = pd.to_datetime(not_cancelled_df_adr['reservation_status_date'])

# Filter for the specified date range
cancelled_df_adr = cancelled_df_adr[(cancelled_df_adr['reservation_status_date'] > '2016-01-01') &
                                    (cancelled_df_adr['reservation_status_date'] < '2017-09-01')]

not_cancelled_df_adr = not_cancelled_df_adr[(not_cancelled_df_adr['reservation_status_date'] > '2016-01-01') &
                                            (not_cancelled_df_adr['reservation_status_date'] < '2017-09-01')]

#ref.chatgpt


# In[34]:


plt.figure(figsize=(15, 7))
plt.title("Average Daily Rate",fontsize=30)

# Plotting ADR for not canceled bookings
plt.plot(not_cancelled_df_adr["reservation_status_date"], not_cancelled_df_adr['adr'], label='not cancelled')

# Plotting ADR for canceled bookings
plt.plot(cancelled_df_adr['reservation_status_date'], cancelled_df_adr['adr'], label='cancelled')

plt.legend(fontsize=20)
plt.show()


# # END
