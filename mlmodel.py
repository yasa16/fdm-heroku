#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
#sns.set_style('darkgrid')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Loading the train data
df_train = pd.read_csv('C:/Users/yasas/Downloads/Train_dataa.csv')

# Looking top 10 rows
df_train.head(10)


# In[4]:


# Looking the bigger picture
df_train.info()


# In[5]:


# Checking the number of missing values in each column
df_train.isnull().sum()


# In[6]:


# Loading the train data
df_test = pd.read_csv('C:/Users/yasas/Downloads/Test_dataa.csv')

# Looking top 10 rows
df_test.head(10)


# In[7]:


# Looking the bigger picture
df_test.info()


# In[8]:


# Checking the number of missing values in each column
df_test.isnull().sum()


# In[9]:



# Removing all those rows that have 3 or more missing values
df_train = df_train.loc[df_train.isnull().sum(axis=1)<3]


# In[10]:


# Looking random 10 rows of the data
df_train.sample(10)


# In[11]:



# Removing all those rows that have 3 or more missing values
df_test = df_test.loc[df_test.isnull().sum(axis=1)<3]


# In[12]:


# Looking random 10 rows of the data
df_test.sample(10)


# In[13]:


print('The count of each category\n',df_train.Var_1.value_counts())


# In[14]:


# Checking for null values
df_train.Var_1.isnull().sum()


# In[15]:



# Filling the missing values w.r.t other attributes underlying pattern 
df_train.loc[ (pd.isnull(df_train['Var_1'])) & (df_train['Graduated'] == 'Yes'),"Var_1"] = 'Cat_6'
df_train.loc[ (pd.isnull(df_train['Var_1'])) & (df_train['Graduated'] == 'No'),"Var_1"] = 'Cat_4'
df_train.loc[ (pd.isnull(df_train["Var_1"])) & ((df_train['Profession'] == 'Lawyer') | (df_train['Profession'] == 'Artist')),"Var_1"] = 'Cat_6'
df_train.loc[ (pd.isnull(df_train["Var_1"])) & (df_train['Age'] > 40),"Var_1"] = 'Cat_6'


# In[16]:


# Checking for null values
df_test.Var_1.isnull().sum()


# In[17]:


numerical_columns_train=['ID','Age','Work_Experience','Family_Size' ]

categorical_columns_train=['Gender','Ever_Married','Graduated','Profession','Spending_Score','Var_1','Segmentation']


# In[18]:


numerical_columns_test=['ID','Age','Work_Experience','Family_Size' ]

categorical_columns_test=['Gender','Ever_Married','Graduated','Profession','Spending_Score','Var_1']


# In[19]:


# Counting Var_1 in each segment
ax1 = df_train.groupby(["Segmentation"])["Var_1"].value_counts().unstack().round(3)


# Percentage of category of Var_1 in each segment
ax2 = df_train.pivot_table(columns='Var_1',index='Segmentation',values='ID',aggfunc='count')
ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(2)

#count plot
fig, ax = plt.subplots(1,2)
ax1.plot(kind="bar",ax = ax[0],figsize = (15,4))
ax[0].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[0].set_title(str(ax1))

#stacked bars
ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (15,4))
ax[1].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[1].set_title(str(ax2))
plt.show()


# In[20]:


print('The count of gender\n',df_train.Gender.value_counts())


# In[21]:


# Checking the count of missing values
df_train.Gender.isnull().sum()


# In[22]:


# Checking the count of missing values
df_test.Gender.isnull().sum()


# In[23]:


# Counting male-female in each segment
ax1 = df_train.groupby(["Segmentation"])["Gender"].value_counts().unstack().round(3)

# Percentage of male-female in each segment
ax2 = df_train.pivot_table(columns='Gender',index='Segmentation',values='ID',aggfunc='count')
ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(2)

#count plot
fig, ax = plt.subplots(1,2)
ax1.plot(kind="bar",ax = ax[0],figsize = (15,4))
ax[0].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[0].set_title(str(ax1))

#stacked bars
ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (15,4))
ax[1].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[1].set_title(str(ax2))
plt.show()


# In[24]:


print('Count of married vs not married\n',df_train.Ever_Married.value_counts())


# In[25]:


# Checking the count of missing values
df_train.Ever_Married.isnull().sum()


# In[26]:


# Filling the missing values w.r.t other attributes underlying pattern
df_train.loc[ (pd.isnull(df_train["Ever_Married"])) & ((df_train['Spending_Score'] == 'Average') | (df_train['Spending_Score'] == 'High')),"Ever_Married"] = 'Yes'
df_train.loc[ (pd.isnull(df_train["Ever_Married"])) & (df_train['Spending_Score'] == 'Low'),"Ever_Married"] = 'No'
df_train.loc[ (pd.isnull(df_train["Ever_Married"])) & (df_train['Age'] > 40),"Ever_Married"] = 'Yes'
df_train.loc[ (pd.isnull(df_train["Ever_Married"])) & (df_train['Profession'] == 'Healthcare'),"Ever_Married"] = 'No'


# In[27]:


# Checking the count of missing values
df_test.Ever_Married.isnull().sum()


# In[28]:


# Filling the missing values w.r.t other attributes underlying pattern
df_test.loc[ (pd.isnull(df_test["Ever_Married"])) & ((df_test['Spending_Score'] == 'Average') | (df_test['Spending_Score'] == 'High')),"Ever_Married"] = 'Yes'
df_test.loc[ (pd.isnull(df_test["Ever_Married"])) & (df_test['Spending_Score'] == 'Low'),"Ever_Married"] = 'No'
df_test.loc[ (pd.isnull(df_test["Ever_Married"])) & (df_test['Age'] > 40),"Ever_Married"] = 'Yes'
df_test.loc[ (pd.isnull(df_test["Ever_Married"])) & (df_test['Profession'] == 'Healthcare'),"Ever_Married"] = 'No'


# In[29]:


# Counting married and non-married in each segment
ax1 = df_train.groupby(["Segmentation"])["Ever_Married"].value_counts().unstack().round(3)

# Percentage of married and non-married in each segment
ax2 = df_train.pivot_table(columns='Ever_Married',index='Segmentation',values='ID',aggfunc='count')
ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(2)

#count plot
fig, ax = plt.subplots(1,2)
ax1.plot(kind="bar",ax = ax[0],figsize = (15,4))
ax[0].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[0].set_title(str(ax1))

#stacked bars
ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (15,4))
ax[1].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[1].set_title(str(ax2))
plt.show()


# In[30]:


df_train.Age.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])


# In[31]:


# Checking the count of missing values
df_train.Age.isnull().sum()


# In[32]:


# Checking the count of missing values
df_test.Age.isnull().sum()


# In[33]:


# Looking the distribution of column Age
plt.figure(figsize=(10,5))

skewness = round(df_train.Age.skew(),2)
kurtosis = round(df_train.Age.kurtosis(),2)
mean = round(np.mean(df_train.Age),0)
median = np.median(df_train.Age)

plt.subplot(1,2,1)
sns.boxplot(y=df_train.Age)
plt.title('Boxplot\n Mean:{}\n Median:{}\n Skewness:{}\n Kurtosis:{}'.format(mean,median,skewness,kurtosis))

plt.subplot(1,2,2)
sns.distplot(df_train.Age)
plt.title('Distribution Plot\n Mean:{}\n Median:{}\n Skewness:{}\n Kurtosis:{}'.format(mean,median,skewness,kurtosis))

plt.show()


# In[34]:


# Looking the distribution of column Age w.r.t to each segment
a = df_train[df_train.Segmentation =='A']["Age"]
b = df_train[df_train.Segmentation =='B']["Age"]
c = df_train[df_train.Segmentation =='C']["Age"]
d = df_train[df_train.Segmentation =='D']["Age"]

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
sns.boxplot(data = df_train, x = "Segmentation", y="Age")
plt.title('Boxplot')

plt.subplot(1,2,2)
sns.kdeplot(a,shade= False, label = 'A', color = 'yellow')
sns.kdeplot(b,shade= False, label = 'B', color = 'green')
sns.kdeplot(c,shade= False, label = 'C', color = 'red')
sns.kdeplot(d,shade= False, label = 'D', color = 'blue')
plt.xlabel('Age')
plt.ylabel('Density')
plt.title("Mean\n A: {}\n B: {}\n C: {}\n D: {}".format(round(a.mean(),0),round(b.mean(),0),round(c.mean(),0),round(d.mean(),0)))

plt.show()


# In[35]:


# Converting the datatype from float to int
df_train['Age'] = df_train['Age'].astype(int)


# In[36]:


df_train.Age.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])


# In[37]:


# Divide people in the 4 age group
df_train['Age_Bin'] = pd.cut(df_train.Age,bins=[17,30,45,60,90],labels=['17-30','31-45','46-60','60+'])


# In[38]:


# Divide people in the 4 age group
df_test['Age_Bin'] = pd.cut(df_test.Age,bins=[17,30,45,60,90],labels=['17-30','31-45','46-60','60+'])


# In[39]:


# Counting different age group in each segment
ax1 = df_train.groupby(["Segmentation"])["Age_Bin"].value_counts().unstack().round(3)

# Percentage of age bins in each segment
ax2 = df_train.pivot_table(columns='Age_Bin',index='Segmentation',values='ID',aggfunc='count')
ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(2)

#count plot
fig, ax = plt.subplots(1,2)
ax1.plot(kind="bar",ax = ax[0],figsize = (15,4))
ax[0].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[0].set_title(str(ax1))

#stacked bars
ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (15,4))
ax[1].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[1].set_title(str(ax2))
plt.show()


# In[40]:


print('Count of each graduate and non-graduate\n',df_train.Graduated.value_counts())


# In[41]:


# Checking the count of missing values
df_train.Graduated.isnull().sum()


# In[42]:


# Filling the missing values w.r.t other attributes underlying pattern
df_train.loc[ (pd.isnull(df_train["Graduated"])) & (df_train['Spending_Score'] == 'Average'),"Graduated"] = 'Yes'
df_train.loc[ (pd.isnull(df_train["Graduated"])) & (df_train['Profession'] == 'Artist'),"Graduated"] = 'Yes'
df_train.loc[ (pd.isnull(df_train["Graduated"])) & (df_train['Age'] > 49),"Graduated"] = 'Yes'
df_train.loc[ (pd.isnull(df_train["Graduated"])) & (df_train['Var_1'] == 'Cat_4'),"Graduated"] = 'No'
df_train.loc[ (pd.isnull(df_train["Graduated"])) & (df_train['Ever_Married'] == 'Yes'),"Graduated"] = 'Yes'

# Replacing remaining NaN with previous values
df_train['Graduated'] = df_train['Graduated'].fillna(method='pad')


# In[43]:


# Checking the count of missing values
df_test.Graduated.isnull().sum()


# In[44]:


# Filling the missing values w.r.t other attributes underlying pattern
df_test.loc[ (pd.isnull(df_test["Graduated"])) & (df_test['Spending_Score'] == 'Average'),"Graduated"] = 'Yes'
df_test.loc[ (pd.isnull(df_test["Graduated"])) & (df_test['Profession'] == 'Artist'),"Graduated"] = 'Yes'
df_test.loc[ (pd.isnull(df_test["Graduated"])) & (df_test['Age'] > 49),"Graduated"] = 'Yes'
df_test.loc[ (pd.isnull(df_test["Graduated"])) & (df_test['Var_1'] == 'Cat_4'),"Graduated"] = 'No'
df_test.loc[ (pd.isnull(df_test["Graduated"])) & (df_test['Ever_Married'] == 'Yes'),"Graduated"] = 'Yes'

# Replacing remaining NaN with previous values
df_test['Graduated'] = df_test['Graduated'].fillna(method='pad')


# In[45]:


# Counting graduate and non-graduate in each segment
ax1 = df_train.groupby(["Segmentation"])["Graduated"].value_counts().unstack().round(3)

# Percentage of graduate and non-graduate in each segment
ax2 = df_train.pivot_table(columns='Graduated',index='Segmentation',values='ID',aggfunc='count')
ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(2)

#count plot
fig, ax = plt.subplots(1,2)
ax1.plot(kind="bar",ax = ax[0],figsize = (15,4))
ax[0].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[0].set_title(str(ax1))

#stacked bars
ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (15,4))
ax[1].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[1].set_title(str(ax2))
plt.show()


# In[46]:


print('Count of each profession\n',df_train.Profession.value_counts())


# In[47]:


# Checking the count of missing values
df_train.Profession.isnull().sum()


# In[48]:


# Filling the missing values w.r.t other attributes underlying pattern
df_train.loc[ (pd.isnull(df_train["Profession"])) & (df_train['Work_Experience'] > 8),"Profession"] = 'Homemaker'
df_train.loc[ (pd.isnull(df_train["Profession"])) & (df_train['Age'] > 70),"Profession"] = 'Lawyer'
df_train.loc[ (pd.isnull(df_train["Profession"])) & (df_train['Family_Size'] < 3),"Profession"] = 'Lawyer'
df_train.loc[ (pd.isnull(df_train["Profession"])) & (df_train['Spending_Score'] == 'Average'),"Profession"] = 'Artist'
df_train.loc[ (pd.isnull(df_train["Profession"])) & (df_train['Graduated'] == 'Yes'),"Profession"] = 'Artist'
df_train.loc[ (pd.isnull(df_train["Profession"])) & (df_train['Ever_Married'] == 'Yes'),"Profession"] = 'Artist'
df_train.loc[ (pd.isnull(df_train["Profession"])) & (df_train['Ever_Married'] == 'No'),"Profession"] = 'Healthcare'
df_train.loc[ (pd.isnull(df_train["Profession"])) & (df_train['Spending_Score'] == 'High'),"Profession"] = 'Executives'


# In[49]:


# Checking the count of missing values
df_test.Profession.isnull().sum()


# In[50]:


# Filling the missing values w.r.t other attributes underlying pattern
df_test.loc[ (pd.isnull(df_test["Profession"])) & (df_test['Work_Experience'] > 8),"Profession"] = 'Homemaker'
df_test.loc[ (pd.isnull(df_test["Profession"])) & (df_test['Age'] > 70),"Profession"] = 'Lawyer'
df_test.loc[ (pd.isnull(df_test["Profession"])) & (df_test['Family_Size'] < 3),"Profession"] = 'Lawyer'
df_test.loc[ (pd.isnull(df_test["Profession"])) & (df_test['Spending_Score'] == 'Average'),"Profession"] = 'Artist'
df_test.loc[ (pd.isnull(df_test["Profession"])) & (df_test['Graduated'] == 'Yes'),"Profession"] = 'Artist'
df_test.loc[ (pd.isnull(df_test["Profession"])) & (df_test['Ever_Married'] == 'Yes'),"Profession"] = 'Artist'
df_test.loc[ (pd.isnull(df_test["Profession"])) & (df_test['Ever_Married'] == 'No'),"Profession"] = 'Healthcare'
df_test.loc[ (pd.isnull(df_test["Profession"])) & (df_test['Spending_Score'] == 'High'),"Profession"] = 'Executives'


# In[51]:


# Count of segments in each profession
ax1 = df_train.groupby(["Profession"])["Segmentation"].value_counts().unstack().round(3)

# Percentage of segments in each profession
ax2 = df_train.pivot_table(columns='Segmentation',index='Profession',values='ID',aggfunc='count')
ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(2)

#count plot
fig, ax = plt.subplots(1,2)
ax1.plot(kind="bar",ax = ax[0],figsize = (16,5))
label = ['Artist','Doctor','Engineer','Entertainment','Executives','Healthcare','Homemaker','Lawyer','Marketing']
ax[0].set_xticklabels(labels = label,rotation = 45)

#stacked bars
ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (16,5))
ax[1].set_xticklabels(labels = label,rotation = 45)

plt.show()


# In[52]:


df_train.Work_Experience.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])


# In[53]:


# Checking the count of missing values
df_train.Work_Experience.isnull().sum()


# In[54]:


# Replacing NaN with previous values
df_train['Work_Experience'] = df_train['Work_Experience'].fillna(method='pad')


# In[55]:


# Checking the count of missing values
df_test.Work_Experience.isnull().sum()


# In[56]:


# Replacing NaN with previous values
df_test['Work_Experience'] = df_test['Work_Experience'].fillna(method='pad')


# In[57]:


# Looking the distribution of column Work Experience
plt.figure(figsize=(15,10))

skewness = round(df_train.Work_Experience.skew(),2)
kurtosis = round(df_train.Work_Experience.kurtosis(),2)
mean = round(np.mean(df_train.Work_Experience),0)
median = np.median(df_train.Work_Experience)

plt.subplot(1,2,1)
sns.boxplot(y=df_train.Work_Experience)
plt.title('Boxplot\n Mean:{}\n Median:{}\n Skewness:{}\n Kurtosis:{}'.format(mean,median,skewness,kurtosis))

plt.subplot(2,2,2)
sns.distplot(df_train.Work_Experience)
plt.title('Distribution Plot\n Mean:{}\n Median:{}\n Skewness:{}\n Kurtosis:{}'.format(mean,median,skewness,kurtosis))

plt.show()


# In[58]:


# Looking the distribution of column Work_Experience w.r.t to each segment
a = df_train[df_train.Segmentation =='A']["Work_Experience"]
b = df_train[df_train.Segmentation =='B']["Work_Experience"]
c = df_train[df_train.Segmentation =='C']["Work_Experience"]
d = df_train[df_train.Segmentation =='D']["Work_Experience"]

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
sns.boxplot(data = df_train, x = "Segmentation", y="Work_Experience")
plt.title('Boxplot')

plt.subplot(1,2,2)
sns.kdeplot(a,shade= False, label = 'A')
sns.kdeplot(b,shade= False, label = 'B')
sns.kdeplot(c,shade= False, label = 'C')
sns.kdeplot(d,shade= False, label = 'D')
plt.xlabel('Work Experience')
plt.ylabel('Density')
plt.title("Mean\n A: {}\n B: {}\n C: {}\n D: {}".format(round(a.mean(),0),round(b.mean(),0),round(c.mean(),0),round(d.mean(),0)))

plt.show()


# In[59]:


# Dividing the people into 3 category of work experience 
df_train['Work_Exp_Category'] = pd.cut(df_train.Work_Experience,bins=[-1,1,7,15],labels=['Low Experience','Medium Experience','High Experience'])


# In[60]:


# Counting different category of work experience in each segment
ax1 = df_train.groupby(["Segmentation"])["Work_Exp_Category"].value_counts().unstack().round(3)

# Percentage of work experience in each segment
ax2 = df_train.pivot_table(columns='Work_Exp_Category',index='Segmentation',values='ID',aggfunc='count')
ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(2)

#count plot
fig, ax = plt.subplots(1,2)
ax1.plot(kind="bar",ax = ax[0],figsize = (15,4))
ax[0].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[0].set_title(str(ax1))

#stacked bars
ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (15,4))
ax[1].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[1].set_title(str(ax2))
plt.show()


# In[61]:


# Dividing the people into 3 category of work experience 
df_test['Work_Exp_Category'] = pd.cut(df_train.Work_Experience,bins=[-1,1,7,15],labels=['Low Experience','Medium Experience','High Experience'])


# In[62]:


print('Count of spending score\n',df_train.Spending_Score.value_counts())


# In[63]:


# Checking the count of missing values
df_train.Spending_Score.isnull().sum()


# In[64]:


# Checking the count of missing values
df_test.Spending_Score.isnull().sum()


# In[65]:


# Counting different category of spending score in each segment
ax1 = df_train.groupby(["Segmentation"])["Spending_Score"].value_counts().unstack().round(3)

# Percentage of spending score in each segment
ax2 = df_train.pivot_table(columns='Spending_Score',index='Segmentation',values='ID',aggfunc='count')
ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(2)

#count plot
fig, ax = plt.subplots(1,2)
ax1.plot(kind="bar",ax = ax[0],figsize = (15,4))
ax[0].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[0].set_title(str(ax1))

#stacked bars
ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (15,4))
ax[1].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[1].set_title(str(ax2))
plt.show()


# In[66]:


df_train.Family_Size.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])


# In[67]:


# Checking the count of missing values
df_train.Family_Size.isnull().sum()


# In[68]:


# Filling the missing values w.r.t other attributes underlying pattern
df_train.loc[ (pd.isnull(df_train["Family_Size"])) & (df_train['Ever_Married'] == 'Yes'),"Family_Size"] = 2.0
df_train.loc[ (pd.isnull(df_train["Family_Size"])) & (df_train['Var_1'] == 'Cat_6'),"Family_Size"] = 2.0
df_train.loc[ (pd.isnull(df_train["Family_Size"])) & (df_train['Graduated'] == 'Yes'),"Family_Size"] = 2.0

# Fill remaining NaN with previous values
df_train['Family_Size'] = df_train['Family_Size'].fillna(method='pad')


# In[69]:


# Checking the count of missing values
df_test.Family_Size.isnull().sum()


# In[70]:


# Filling the missing values w.r.t other attributes underlying pattern
df_test.loc[ (pd.isnull(df_test["Family_Size"])) & (df_test['Ever_Married'] == 'Yes'),"Family_Size"] = 2.0
df_test.loc[ (pd.isnull(df_test["Family_Size"])) & (df_test['Var_1'] == 'Cat_6'),"Family_Size"] = 2.0
df_test.loc[ (pd.isnull(df_test["Family_Size"])) & (df_test['Graduated'] == 'Yes'),"Family_Size"] = 2.0

# Fill remaining NaN with previous values
df_test['Family_Size'] = df_test['Family_Size'].fillna(method='pad')


# In[71]:


# Looking the distribution of column Work Experience
plt.figure(figsize=(15,10))

skewness = round(df_train.Family_Size.skew(),2)
kurtosis = round(df_train.Family_Size.kurtosis(),2)
mean = round(np.mean(df_train.Family_Size),0)
median = np.median(df_train.Family_Size)

plt.subplot(1,2,1)
sns.boxplot(y=df_train.Family_Size)
plt.title('Boxplot\n Mean:{}\n Median:{}\n Skewness:{}\n Kurtosis:{}'.format(mean,median,skewness,kurtosis))

plt.subplot(2,2,2)
sns.distplot(df_train.Family_Size)
plt.title('Distribution Plot\n Mean:{}\n Median:{}\n Skewness:{}\n Kurtosis:{}'.format(mean,median,skewness,kurtosis))

plt.show()


# In[72]:


# Looking the distribution of column Family Size w.r.t to each segment
a = df_train[df_train.Segmentation =='A']["Family_Size"]
b = df_train[df_train.Segmentation =='B']["Family_Size"]
c = df_train[df_train.Segmentation =='C']["Family_Size"]
d = df_train[df_train.Segmentation =='D']["Family_Size"]

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
sns.boxplot(data = df_train, x = "Segmentation", y="Family_Size")
plt.title('Boxplot')

plt.subplot(1,2,2)
sns.kdeplot(a,shade= False, label = 'A')
sns.kdeplot(b,shade= False, label = 'B')
sns.kdeplot(c,shade= False, label = 'C')
sns.kdeplot(d,shade= False, label = 'D')
plt.xlabel('Family Size')
plt.ylabel('Density')
plt.title("Mean\n A: {}\n B: {}\n C: {}\n D: {}".format(round(a.mean(),0),round(b.mean(),0),round(c.mean(),0),round(d.mean(),0)))

plt.show()


# In[73]:


# Changing the data type
df_train['Family_Size'] = df_train['Family_Size'].astype(int)


# In[74]:


df_train.Family_Size.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])


# In[75]:


# Divide family size into 3 category
df_train['Family_Size_Category'] = pd.cut(df_train.Family_Size,bins=[0,4,6,10],labels=['Small Family','Big Family','Joint Family'])


# In[76]:


# Changing the data type
df_test['Family_Size'] = df_test['Family_Size'].astype(int)


# In[77]:


df_test.Family_Size.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])


# In[78]:


# Divide family size into 3 category
df_test['Family_Size_Category'] = pd.cut(df_test.Family_Size,bins=[0,4,6,10],labels=['Small Family','Big Family','Joint Family'])


# In[79]:


# Counting different category of family size in each segment
ax1 = df_train.groupby(["Segmentation"])["Family_Size_Category"].value_counts().unstack().round(3)

# Percentage of family size in each segment
ax2 = df_train.pivot_table(columns='Family_Size_Category',index='Segmentation',values='ID',aggfunc='count')
ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(3)

#count plot
fig, ax = plt.subplots(1,2)
ax1.plot(kind="bar",ax = ax[0],figsize = (15,4))
ax[0].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[0].set_title(str(ax1))

#stacked bars
ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (15,4))
ax[1].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[1].set_title(str(ax2))
plt.show()


# In[80]:


print('Count of each category of segmentation\n',df_train.Segmentation.value_counts())


# In[81]:


segments = df_train.loc[:,"Segmentation"].value_counts()
plt.xlabel("Segment")
plt.ylabel('Count')
sns.barplot(segments.index , segments.values).set_title('Segments')
plt.show()


# In[82]:


df_train.reset_index(drop=True, inplace=True)
df_train.info()


# In[83]:


# number of unique ids
df_train.ID.nunique()


# In[84]:


# number of unique ids in Test Set
df_test.ID.nunique()


# In[85]:


df_train.describe(include='all')


# In[86]:


df_train = df_train[['ID','Gender', 'Ever_Married', 'Age', 'Age_Bin', 'Graduated', 'Profession', 'Work_Experience', 'Work_Exp_Category',
         'Spending_Score', 'Family_Size', 'Family_Size_Category','Var_1', 'Segmentation']]
df_train.head(10)


# In[87]:


df1 = df_train.copy()
df1.head()


# In[88]:


# Separating dependent-independent variables
X = df1.drop('Segmentation',axis=1)
y = df1['Segmentation']


# In[89]:


# import the train-test split
from sklearn.model_selection import train_test_split

# divide into train and test sets
df1_trainX, df1_testX, df1_trainY, df1_testY = train_test_split(X,y, train_size = 0.7, random_state = 101, stratify=y)


# In[90]:


# converting binary variables to numeric
df1_trainX['Gender'] = df1_trainX['Gender'].replace(('Male','Female'),(1,0))
df1_trainX['Ever_Married'] = df1_trainX['Ever_Married'].replace(('Yes','No'),(1,0))
df1_trainX['Graduated'] = df1_trainX['Graduated'].replace(('Yes','No'),(1,0))
df1_trainX['Spending_Score'] = df1_trainX['Spending_Score'].replace(('High','Average','Low'),(3,2,1))

# converting nominal variables into dummy variables
pf = pd.get_dummies(df1_trainX.Profession,prefix='Profession')
df1_trainX = pd.concat([df1_trainX,pf],axis=1)

vr = pd.get_dummies(df1_trainX.Var_1,prefix='Var_1')
df1_trainX = pd.concat([df1_trainX,vr],axis=1)

# scaling continuous variables
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1_trainX[['Age','Work_Experience','Family_Size']] = scaler.fit_transform(df1_trainX[['Age','Work_Experience','Family_Size']])

df1_trainX.drop(['ID','Age_Bin','Profession','Work_Exp_Category','Family_Size_Category','Var_1'], axis=1, inplace=True)


# In[91]:


# converting binary variables to numeric
df1_testX['Gender'] = df1_testX['Gender'].replace(('Male','Female'),(1,0))
df1_testX['Ever_Married'] = df1_testX['Ever_Married'].replace(('Yes','No'),(1,0))
df1_testX['Graduated'] = df1_testX['Graduated'].replace(('Yes','No'),(1,0))
df1_testX['Spending_Score'] = df1_testX['Spending_Score'].replace(('High','Average','Low'),(3,2,1))

# converting nominal variables into dummy variables
pf = pd.get_dummies(df1_testX.Profession,prefix='Profession')
df1_testX = pd.concat([df1_testX,pf],axis=1)

vr = pd.get_dummies(df1_testX.Var_1,prefix='Var_1')
df1_testX = pd.concat([df1_testX,vr],axis=1)

# scaling continuous variables
df1_testX[['Age','Work_Experience','Family_Size']] = scaler.transform(df1_testX[['Age','Work_Experience','Family_Size']])

df1_testX.drop(['ID','Age_Bin','Profession','Work_Exp_Category','Family_Size_Category','Var_1'], axis=1, inplace=True)


# In[92]:


df1_trainX.shape, df1_trainY.shape, df1_testX.shape, df1_testY.shape


# In[93]:


# Correlation matrix
plt.figure(figsize=(17,10))
sns.heatmap(df1_trainX.corr(method='spearman').round(2),linewidth = 0.5,annot=True,cmap="YlGnBu")
plt.show()


# In[94]:


df2 = df_train.copy()
df2.head()


# In[95]:


# Separating dependent-independent variables
X = df2.drop('Segmentation',axis=1)
y = df2['Segmentation']


# In[96]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier
selector = ExtraTreesClassifier(random_state = 42)
selector.fit(df1_trainX, df1_trainY)
feature_imp = selector.feature_importances_
for index, val in enumerate(feature_imp):
    print(index, round((val * 100), 2))


# In[97]:


# import the train-test split
from sklearn.model_selection import train_test_split

# divide into train and test sets
df2_trainX, df2_testX, df2_trainY, df2_testY = train_test_split(X,y, train_size = 0.7, random_state = 101, stratify=y)


# In[98]:


# Converting binary to numeric
df2_trainX['Gender'] = df2_trainX['Gender'].replace(('Male','Female'),(1,0))
df2_trainX['Ever_Married'] = df2_trainX['Ever_Married'].replace(('Yes','No'),(1,0))
df2_trainX['Graduated'] = df2_trainX['Graduated'].replace(('Yes','No'),(1,0))

# Converting nominal variables to dummy variables
ab = pd.get_dummies(df2_trainX.Age_Bin,prefix='Age_Bin')
df2_trainX = pd.concat([df2_trainX,ab],axis=1)

pf = pd.get_dummies(df2_trainX.Profession,prefix='Profession')
df2_trainX = pd.concat([df2_trainX,pf],axis=1)

we = pd.get_dummies(df2_trainX.Work_Exp_Category,prefix='WorkExp')
df2_trainX = pd.concat([df2_trainX,we],axis=1)

sc = pd.get_dummies(df2_trainX.Spending_Score,prefix='Spending')
df2_trainX = pd.concat([df2_trainX,sc],axis=1)


fs = pd.get_dummies(df2_trainX.Family_Size_Category,prefix='FamilySize')
df2_trainX = pd.concat([df2_trainX,fs],axis=1)

vr = pd.get_dummies(df2_trainX.Var_1,prefix='Var_1')
df2_trainX = pd.concat([df2_trainX,vr],axis=1)

df2_trainX.drop(['ID','Age','Age_Bin','Profession','Work_Experience','Work_Exp_Category','Spending_Score',
               'Family_Size','Family_Size_Category','Var_1'],axis=1,inplace=True)


# In[99]:


# Converting binary to numeric
df2_testX['Gender'] = df2_testX['Gender'].replace(('Male','Female'),(1,0))
df2_testX['Ever_Married'] = df2_testX['Ever_Married'].replace(('Yes','No'),(1,0))
df2_testX['Graduated'] = df2_testX['Graduated'].replace(('Yes','No'),(1,0))

# Converting nominal variables to dummy variables
ab = pd.get_dummies(df2_testX.Age_Bin,prefix='Age_Bin')
df2_testX = pd.concat([df2_testX,ab],axis=1)

pf = pd.get_dummies(df2_testX.Profession,prefix='Profession')
df2_testX = pd.concat([df2_testX,pf],axis=1)

we = pd.get_dummies(df2_testX.Work_Exp_Category,prefix='WorkExp')
df2_testX = pd.concat([df2_testX,we],axis=1)

sc = pd.get_dummies(df2_testX.Spending_Score,prefix='Spending')
df2_testX = pd.concat([df2_testX,sc],axis=1)


fs = pd.get_dummies(df2_testX.Family_Size_Category,prefix='FamilySize')
df2_testX = pd.concat([df2_testX,fs],axis=1)

vr = pd.get_dummies(df2_testX.Var_1,prefix='Var_1')
df2_testX = pd.concat([df2_testX,vr],axis=1)

df2_testX.drop(['ID','Age','Age_Bin','Profession','Work_Experience','Work_Exp_Category','Spending_Score',
               'Family_Size','Family_Size_Category','Var_1'],axis=1,inplace=True)


# In[100]:


df2_trainX.shape, df2_trainY.shape, df2_testX.shape, df2_testY.shape


# In[101]:


# Correlation matrix
plt.figure(figsize=(17,10))
sns.heatmap(df2_trainX.corr(method='spearman').round(2),linewidth = 0.5,annot=True,cmap="YlGnBu")
plt.show()


# In[102]:


#MODEL BUILDING
#Decision Tree
train_dt1_x = df1_trainX.copy()
train_dt1_x.head()


# In[103]:


train_dt1_y = df1_trainY.copy()
train_dt1_y.head()


# In[104]:


# importing decision tree classifier 
from sklearn.tree import DecisionTreeClassifier

# creating the decision tree function
model_dt1 = DecisionTreeClassifier(random_state=10,criterion='gini')

#fitting the model
model_dt1.fit(train_dt1_x, train_dt1_y)

# depth of the decision tree
print('Depth of the Decision Tree: ', model_dt1.get_depth())

#checking the training score
print('Accuracy on training: ',model_dt1.score(train_dt1_x, train_dt1_y))

# predict the target on the train dataset
yhat1 = model_dt1.predict(train_dt1_x)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(train_dt1_y.values, yhat1, labels=["A","B","C","D"])
print('-------The confusion matrix for this model is-------')
print(cm1)

from sklearn.metrics import classification_report
print('\n\n-------Printing the whole report of the model-------')
print(classification_report(train_dt1_y.values, yhat1))


# In[105]:


X1 = train_dt1_x.copy()
y1 = pd.DataFrame({'Seg':train_dt1_y})
y1['Seg'] = y1['Seg'].replace(('A','B','C','D'),(1,2,3,4))


# In[106]:


# Implementing grid search

parameter_grid = {
    'max_depth' : [24,25,26,27,28,29,30],
    'max_features': [0.3, 0.5, 0.7]
    }

from sklearn.model_selection import GridSearchCV
gridsearch = GridSearchCV(estimator=model_dt1, param_grid=parameter_grid, scoring='neg_mean_squared_error', cv=5)

gridsearch.fit(X1, y1)

print(gridsearch.best_params_)


# In[107]:


# Implementing random search

parameter_grid = {
    'max_depth' : [24,25,26,27,28,29,30],
    'max_features': [0.3, 0.5, 0.7,0.9]
    }

from sklearn.model_selection import RandomizedSearchCV

randomsearch = RandomizedSearchCV(estimator=model_dt1, param_distributions=parameter_grid, n_iter= 10, cv=5)
randomsearch.fit(X1, y1)

print(randomsearch.best_params_)


# In[108]:


# final model
model_dt1 = DecisionTreeClassifier(max_depth=26, max_features=0.9 ,random_state=10)

# fitting the model
model_dt1.fit(train_dt1_x, train_dt1_y)

# Training score
print(model_dt1.score(train_dt1_x, train_dt1_y).round(4))


# In[109]:


from sklearn import tree

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(model_dt1, feature_names=train_dt1_x.columns, max_depth=2, filled=True)


# In[110]:


#Predicting on test set
test_dt1_x = df1_testX.copy()
test_dt1_x.head()


# In[111]:


test_dt1_y = df1_testY.copy()
test_dt1_y.head()


# In[112]:


y_dt1 = model_dt1.predict(test_dt1_x)
y_dt1


# In[113]:


from sklearn.metrics import confusion_matrix
print('-------The confusion matrix for test data is-------\n')
print(confusion_matrix(test_dt1_y.values, y_dt1, labels=["A","B","C","D"]))

from sklearn.metrics import classification_report
print('\n\n-------Printing the report of test data-------\n')
print(classification_report(test_dt1_y.values, y_dt1))


# In[114]:


pd.Series(y_dt1).value_counts()


# In[115]:


#Building the model with second type of dataframe(df_type2)
train_dt2_x = df2_trainX.copy()
train_dt2_x.head()


# In[116]:


train_dt2_y = df2_trainY.copy()
train_dt2_y.head()


# In[117]:


# importing decision tree classifier 
from sklearn.tree import DecisionTreeClassifier

# creating the decision tree function
model_dt2 = DecisionTreeClassifier(random_state=10,criterion='gini')

#fitting the model
model_dt2.fit(train_dt2_x, train_dt2_y)

# depth of the decision tree
print('Depth of the Decision Tree: ', model_dt2.get_depth())

#checking the training score
print('Accuracy on training: ',model_dt2.score(train_dt2_x, train_dt2_y))

# predict the target on the train dataset
yhat2 = model_dt2.predict(train_dt2_x)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(train_dt2_y.values, yhat2, labels=["A","B","C","D"])
print('-------The confusion matrix for this model is-------')
print(cm2)

from sklearn.metrics import classification_report
print('\n\n-------Printing the whole report of the model-------')
print(classification_report(train_dt2_y.values, yhat2))


# In[118]:


X2 = train_dt2_x.copy()
y2 = pd.DataFrame({'Seg':train_dt2_y})
y2['Seg'] = y2['Seg'].replace(('A','B','C','D'),(1,2,3,4))


# In[119]:


# Implementing grid search

parameter_grid = {
    'max_depth' : [24,25,26,27,28,29,30],
    'max_features': [0.3, 0.5, 0.7]
    }

from sklearn.model_selection import GridSearchCV
gridsearch = GridSearchCV(estimator=model_dt2, param_grid=parameter_grid, scoring='neg_mean_squared_error', cv=5)

gridsearch.fit(X2, y2)

print(gridsearch.best_params_)


# In[120]:


# Implementing random search

parameter_grid = {
    'max_depth' : [24,25,26,27,28,29,30],
    'max_features': [0.3, 0.5, 0.7,0.9]
    }

from sklearn.model_selection import RandomizedSearchCV

randomsearch = RandomizedSearchCV(estimator=model_dt2, param_distributions=parameter_grid, n_iter= 10, cv=5)
randomsearch.fit(X2, y2)

print(randomsearch.best_params_)


# In[121]:


# final model
model_dt2 = DecisionTreeClassifier(max_depth=25, max_features=0.7, random_state=10)

#fitting the model
model_dt2.fit(train_dt2_x, train_dt2_y)

#Training score
print(model_dt2.score(train_dt2_x, train_dt2_y).round(4))


# In[122]:


from sklearn import tree

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(model_dt2, feature_names=train_dt2_x.columns, max_depth=2, filled=True)


# In[123]:


#PREDICTING ON TEST SET
test_dt2_x = df2_testX.copy()
test_dt2_x.head()


# In[124]:


test_dt2_y = df2_testY.copy()
test_dt2_y.head()


# In[125]:


y_dt2 = model_dt2.predict(test_dt2_x)
y_dt2


# In[126]:


from sklearn.metrics import confusion_matrix
print('-------The confusion matrix for test data is-------')
print(confusion_matrix(test_dt2_y.values, y_dt2, labels=["A","B","C","D"]))

from sklearn.metrics import classification_report
print('\n\n-------Printing the report of test data-------')
print(classification_report(test_dt2_y.values, y_dt2))


# In[127]:


pd.Series(y_dt2).value_counts()


# In[128]:


#MODEL EVALUATION
print('************************  MODEL-1 REPORT  *********************************\n')
print('Train data')
print(classification_report(train_dt1_y.values, yhat1))
print('\nTest data')
print(classification_report(test_dt1_y.values, y_dt1))


# In[129]:


print('************************  MODEL-2 REPORT  *********************************\n')
print('Train data')
print(classification_report(train_dt2_y.values, yhat2))
print('\nTest data')
print(classification_report(test_dt2_y.values, y_dt2))


# In[130]:


#RANDOM FOREST MODEL BUILDING
train_rf1_x = df1_trainX.copy()
train_rf1_x.head()


# In[131]:


train_rf1_y = df1_trainY.copy()
train_rf1_y.head()


# In[132]:


# Importing the library
from sklearn.ensemble import RandomForestClassifier

# Instantiate the classifier with 20 decision tree
rfc1 = RandomForestClassifier(random_state=0,n_estimators=20)

# Train model
model_rfc1 = rfc1.fit(train_rf1_x, train_rf1_y)

# Predicting the classes
yhat3 = rfc1.predict(train_rf1_x)

# view the feature scores
feature_scores = pd.Series(rfc1.feature_importances_, index=train_rf1_x.columns).sort_values(ascending=False)
print('The importance of features ranked from high to low:\n',feature_scores)

from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(train_rf1_y.values, yhat3, labels=["A","B","C","D"])
print('\n\n-------The confusion matrix for this model is-------')
print(cm3)

from sklearn.metrics import classification_report
print('\n\n-------Printing the whole report of the model-------')
print(classification_report(train_rf1_y.values, yhat3))


# In[133]:


# Creating bar plot of scores of variables importance
plt.figure(figsize=(10,8))
sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()


# In[134]:


#Predicting on test set
test_rf1_x = df1_testX.copy()
test_rf1_x.head()


# In[135]:


test_rf1_y = df1_testY.copy()
test_rf1_y.head()


# In[136]:


y_rf1 = rfc1.predict(test_rf1_x)
y_rf1


# In[137]:


from sklearn.metrics import confusion_matrix
print('-------The confusion matrix for test data is-------\n')
print(confusion_matrix(test_rf1_y.values, y_rf1, labels=["A","B","C","D"]))

from sklearn.metrics import classification_report
print('\n\n-------Printing the report of test data-------\n')
print(classification_report(test_rf1_y.values, y_rf1))


# In[138]:


pd.Series(y_rf1).value_counts()


# In[139]:


#Building the model with second type of dataframe(df_type2)
train_rf2_x = df2_trainX.copy()
train_rf2_x.head()


# In[140]:


train_rf2_y = df2_trainY.copy()
train_rf2_y.head()


# In[141]:


# Importing the library
#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

# Instantiate the classifier with 20 decision tree
rfc2 = RandomForestClassifier(random_state=0,n_estimators=20)

# Train model
model_rfc2 = rfc2.fit(train_rf2_x, train_rf2_y)

# Predicting the classes
yhat4 = rfc2.predict(train_rf2_x)

# view the feature scores
feature_scores = pd.Series(rfc2.feature_importances_, index=train_rf2_x.columns).sort_values(ascending=False)
print('The importance of features ranked from high to low:\n',feature_scores)

from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(train_rf2_y.values, yhat4, labels=["A","B","C","D"])
print('\n\n-------The confusion matrix for this model is-------')
print(cm4)

from sklearn.metrics import classification_report
print('\n\n-------Printing the whole report of the model-------')
print(classification_report(train_rf2_y.values, yhat4))


# In[142]:


# Creating bar plot of scores of variables importance
plt.figure(figsize=(10,8))
sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()


# In[143]:


#Predicting on test set
test_rf2_x = df2_testX.copy()
test_rf2_x.head()


# In[144]:


test_rf2_y = df2_testY.copy()
test_rf2_y.head()


# In[145]:


y_rf2 = rfc2.predict(test_rf2_x)
y_rf2


# In[146]:


from sklearn.metrics import confusion_matrix
print('-------The confusion matrix for test data is-------\n')
print(confusion_matrix(test_rf2_y.values, y_rf2, labels=["A","B","C","D"]))

from sklearn.metrics import classification_report
print('\n\n-------Printing the report of test data-------\n')
print(classification_report(test_rf2_y.values, y_rf2))


# In[147]:


pd.Series(y_rf2).value_counts()


# In[148]:


#MODEL EVALUATION
print('************************  MODEL-1 REPORT  *********************************\n')
print('Train data')
print(classification_report(train_rf1_y.values, yhat3))
print('\nTest data')
print(classification_report(test_rf1_y.values, y_rf1))


# In[149]:


print('************************  MODEL-2 REPORT  *********************************\n')
print('Train data')
print(classification_report(train_rf2_y.values, yhat4))
print('\nTest data')
print(classification_report(test_rf2_y.values, y_rf2))


# In[150]:


#Make predictions using the features from the test data set
predictions = model_dt1.predict(train_dt1_x[0:2628])
#Display our predictions - they are either 0 or 1 for each training instance
#depending on whether our algorithm believes the person survived or not.
predictions



# In[152]:


# Saving model to disk
import pickle
pickle.dump(model_rfc2, open('model.pkl','wb'))


# In[153]:
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


# In[ ]:




