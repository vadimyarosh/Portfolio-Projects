#!/usr/bin/env python
# coding: utf-8

# In[1]:


#First, I ensure I have necessary Python libraries installed:
pip install pandas matplotlib seaborn scikit-learn


# In[2]:


#I import relevant packages
import pandas as pd 
import numpy as np 
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from IPython.display import display, HTML

#I load the dataset
df=pd.read_csv(r"C:\Users\vadim\Downloads\Housing.csv")

df.head()


# In[3]:


df.columns


# In[4]:


# Function to create scrollable table within a small window
def create_scrollable_table(df, table_id, title):
    html = f'<h3>{title}</h3>'
    html += f'<div id="{table_id}" style="height:200px; overflow:auto;">'
    html += df.to_html()
    html += '</div>'
    return html


# In[5]:


df.shape


# In[6]:


#This line selects numerical columns from our DataFrame and generates descriptive statistics for them, 
#summarizing central tendency, dispersion, and shape of the dataset's distribution.
numerical_features = df.select_dtypes(include=[np.number])
numerical_features.describe()


# In[7]:


# Summary statistics for numerical features. 

numerical_features = df.select_dtypes(include=[np.number])
summary_stats = numerical_features.describe().T
html_numerical = create_scrollable_table(summary_stats, 'numerical_features', 'Summary statistics for numerical features')

display(HTML(html_numerical))


# In[8]:


# Summary statistics for categorical features

categorical_features = df.select_dtypes(include=[object])
cat_summary_stats = categorical_features.describe().T
html_categorical = create_scrollable_table(cat_summary_stats, 'categorical_features', 'Summary statistics for categorical features')

display(HTML(html_categorical ))


# In[9]:


# Null values in the dataset
null_values=df.isnull().sum()
html_null_values = create_scrollable_table(null_values.to_frame(),'null_values','Null values in the dataset')


# In[10]:


# Percentage of missing values for each feature
missing_percentage=(df.isnull().sum()/len(df))/100
html_missing_percentage=create_scrollable_table(missing_percentage.to_frame(),'missing_percentage','Percentage of missing value for each feature')
display(HTML(html_null_values+html_missing_percentage))


# In[11]:


# Convert 'yes'/'no' columns to boolean
yes_no_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in yes_no_columns:
    df[col] = df[col].map({'yes': True, 'no': False})


# In[12]:


print(df.head())


# In[13]:


# Converting 'furnishingstatus' to categorical type and standardizing its values
if 'furnishingstatus' in df.columns: 
    df['furnishingstatus'] = df['furnishingstatus'].astype('category').str.lower()


# In[14]:


print(df['furnishingstatus'].unique())


# In[15]:


# Check for missing values
print(df.isnull().sum())


# In[16]:


#Creating 'total_rooms'
if all(x in df.columns for x in ['bedrooms', 'bathrooms']):  
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']


# In[17]:


df.head()


# In[18]:


duplicates = df.duplicated()


# In[19]:


# To see if there are any duplicates
any_duplicates = duplicates.any()
print(f"Are there any duplicates? {any_duplicates}")


# In[20]:


# If Iwant to see the duplicated rows
if any_duplicates:
    duplicated_rows = df[duplicates]
    print(duplicated_rows)


# In[21]:


#By Size or Feature: Filter by houses with a certain number of bedrooms, bathrooms, 
#or specific amenities like air conditioning to study their impact on price.
houses_with_ac = df[df['airconditioning'] == True]
houses_3_bedrooms = df[df['bedrooms'] == 3]


# In[22]:


houses_with_ac.head()


# In[23]:


houses_3_bedrooms.head()


# In[24]:


# Assuming 'area' greater than 10000 is considered an outlier
df_without_outliers = df[df['area'] <= 10000]


# In[25]:


print(df_without_outliers)


# In[26]:


average_price_by_area = df.groupby('area')['price'].mean()


# In[27]:


print(average_price_by_area)


# In[28]:


median_price_by_bedrooms = df.groupby('bedrooms')['price'].median()

# To see the results
print(median_price_by_bedrooms)


# In[29]:


price_distribution_by_furnishing = df.groupby('furnishingstatus')['price'].describe()

# To see the results
print(price_distribution_by_furnishing)


# In[31]:


# Select only numeric columns from the DataFrame
numeric_df = df.select_dtypes(include=[np.number])

# Now perform correlation analysis on the numeric DataFrame
correlation_matrix = numeric_df.corr()

# Display the correlation matrix
print(correlation_matrix)


# In[32]:


import seaborn as sns
import matplotlib.pyplot as plt

# Visualizing the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Numeric Features')
plt.show()


# In[35]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='area', y='price', alpha=0.6)
plt.title('Price vs. Area')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()


# In[41]:


plt.figure(figsize=(10, 6))
sns.barplot(x='bedrooms', y='price', data=df, estimator=np.mean)
plt.title('Average Price vs. Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Average Price')
plt.show()


# In[44]:


# Setting the aesthetic style of the plots
sns.set(style="whitegrid")

# Creating subplots
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Box plot for air conditioning
sns.boxplot(x='airconditioning', y='price', data=df, ax=ax[0])
ax[0].set_title('Price Distribution with and without Air Conditioning')
ax[0].set_xlabel('Air Conditioning')
ax[0].set_ylabel('Price')

# Box plot for hot water heating
sns.boxplot(x='hotwaterheating', y='price', data=df, ax=ax[1])
ax[1].set_title('Price Distribution with and without Hot Water Heating')
ax[1].set_xlabel('Hot Water Heating')
ax[1].set_ylabel('Price')

plt.tight_layout()
plt.show()


# In[45]:


# Setting the aesthetic style of the plots
sns.set(style="whitegrid")

# Box plot for furnishing status
plt.figure(figsize=(10, 6))
sns.boxplot(x='furnishingstatus', y='price', data=df)
plt.title('Price Distribution by Furnishing Status')
plt.xlabel('Furnishing Status')
plt.ylabel('Price')

plt.show()


# In[46]:


# Setting the aesthetic style of the plots
sns.set(style="whitegrid")

# Creating subplots for the location-based attributes
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Box plot for houses on a main road
sns.boxplot(x='mainroad', y='price', data=df, ax=ax[0])
ax[0].set_title('Price Distribution with and without Main Road Access')
ax[0].set_xlabel('Main Road Access')
ax[0].set_ylabel('Price')

# Box plot for houses in a preferred area
sns.boxplot(x='prefarea', y='price', data=df, ax=ax[1])
ax[1].set_title('Price Distribution in and out of Preferred Area')
ax[1].set_xlabel('Preferred Area')
ax[1].set_ylabel('Price')

plt.tight_layout()
plt.show()


# In[ ]:




