#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing Required Libraries
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# data Reading
Books = pd.read_csv('bestsellers with categories.csv')


# In[3]:


Books.head()


# In[4]:


# know the null values
Books.isnull().sum()


# In[5]:


Books.info()


# # Data Cleaning

# In[6]:


Books.head()


# In[7]:


Books['Name'].unique()


# In[8]:


Books['Name'].nunique()


# In[9]:


Books['Author'].unique()


# In[10]:


Books['Author'].nunique()


# In[11]:


Books['Year'].unique()


# In[12]:


Books['Year'].nunique()


# In[13]:


# sorting the year
Books=Books.sort_values(by=['Year'],ascending=[True])
Books.head()


# In[14]:


Genre=pd.get_dummies(Books.Genre,prefix='Genre')


# In[15]:


Books=Books.join(Genre)


# In[16]:


Books=Books.drop(['Genre'],axis=1)


# In[17]:


Books.head()


# In[18]:


Books=Books.drop(['Genre_Non Fiction'],axis=1)


# In[19]:


Books.head()


# In[27]:


#plotting The Target variable
px.scatter(Books.Year,Books.Genre_Fiction)


# In[31]:


# This Data set follows the supervised Learining algorithams under classification Technique 


# In[34]:


Books.columns


# # Prediuctive models

# # LogisticRegression

# In[32]:


from sklearn.model_selection import train_test_split


# In[40]:


X = Books[['User Rating', 'Reviews', 'Price', 'Year']]
y = Books[['Genre_Fiction']]


# In[41]:


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.80,random_state=123)


# In[42]:


from sklearn.linear_model import LogisticRegression


# In[43]:


model1 = LogisticRegression()


# In[44]:


model1.fit(X_train,y_train)


# In[45]:


from sklearn import metrics


# In[46]:


y_pred = pd.Series(model1.predict(X_test))


# In[47]:


y_pred


# In[48]:


y_test


# In[49]:


y_test = y_test.reset_index(drop=True)


# In[50]:


y_test


# In[51]:


z1 = pd.concat([y_test,y_pred],axis=1)


# In[52]:


z1.columns = ['Actual','Predicted']


# In[53]:


z1.head(15)


# In[54]:


metrics.accuracy_score(y_test,y_pred)


# In[55]:


metrics.confusion_matrix(y_test,y_pred)


# In[56]:


# Logistic Regression is not giving Better Result So i am moivng to decision Tree 


# # DecisionTree

# In[60]:


from sklearn.tree import DecisionTreeClassifier


# In[61]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[62]:


y_pred


# In[63]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[64]:


from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))


# In[65]:


# DecisionTreeClassifier is giving Better to compare with Logistic Regression but i can go to Random Forest


# # Randomforest

# In[66]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[67]:


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[68]:


y_pred = y_pred.astype(int)


# In[72]:


from sklearn.metrics import accuracy_score, classification_report

print(accuracy_score(y_test, y_pred))


print(classification_report(y_test, y_pred))


# In[73]:


# Randomforest is not giving better result to compare with others so i want to try with Support Vector machine


# # SupportVectorMachine

# In[74]:


from sklearn.model_selection import train_test_split


# In[75]:


X = Books[['User Rating', 'Reviews', 'Price', 'Year']]
y = Books[['Genre_Fiction']]


# In[76]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[77]:


len(X_train)


# In[78]:


len(X_test)


# In[81]:


from sklearn.svm import SVC
modelSVC = SVC()


# In[82]:


modelSVC.fit(X_train, y_train)


# In[84]:


modelSVC.score(X_test, y_test)


# # Tune parameters
# 
# 1. Regularization (C)

# In[85]:


model_C = SVC(C=1)
model_C.fit(X_train, y_train)
model_C.score(X_test, y_test)


# In[86]:


model_C = SVC(C=10)
model_C.fit(X_train, y_train)
model_C.score(X_test, y_test)

2. Gamma
# In[87]:


model_g = SVC(gamma=10)
model_g.fit(X_train, y_train)
model_g.score(X_test, y_test)

3.Kernal
# In[88]:


model_linear_kernal = SVC(kernel='linear')
model_linear_kernal.fit(X_train, y_train)


# In[89]:


model_linear_kernal.score(X_test, y_test)


# In[91]:


# After Doing this Process I Made my decision
# DecisionTreeClassifier is good model For this Dataset It provides the heighest Accuracy To comparing with other Techniques


# In[ ]:




