#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[4]:


titanic = pd.read_csv('C:\\Users\\lenovo\\Downloads\\titanic.csv')
titanic


# In[5]:


titanic.head()


# In[6]:


titanic.tail()


# In[7]:


titanic.shape


# In[8]:


titanic.columns


# In[9]:


titanic.dtypes


# In[10]:


titanic.duplicated().sum()


# In[11]:


nv = titanic.isna().sum().sort_values(ascending=False)
nv = nv[nv>0]
nv


# In[12]:


titanic.isnull().sum().sort_values(ascending=False)*100/len(titanic)


# In[13]:


titanic.drop(columns = 'Cabin', axis = 1, inplace = True)
titanic.columns


# In[14]:


titanic['Age'].fillna(titanic['Age'].mean(),inplace=True)

# filling null values in Embarked Column with mode values of embarked column
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0],inplace=True)


# In[15]:


titanic.isna().sum()


# In[16]:


titanic[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Embarked']].nunique().sort_values()


# In[17]:


titanic['Survived'].unique()


# In[18]:


titanic['Sex'].unique()


# In[19]:


titanic['Pclass'].unique()


# In[20]:


titanic['SibSp'].unique()


# In[21]:


titanic['Parch'].unique()


# In[22]:


titanic['Embarked'].unique()


# In[23]:


titanic.drop(columns=['PassengerId','Name','Ticket'],axis=1,inplace=True)
titanic.columns


# In[24]:


titanic.info()


# In[25]:


titanic.describe()


# In[26]:


titanic.describe(include='O')


# In[27]:


d1 = titanic['Sex'].value_counts()
d1


# In[28]:


sns.countplot(x=titanic['Sex'])
plt.show()


# In[29]:


plt.figure(figsize=(5,5))
plt.pie(d1.values,labels=d1.index,autopct='%.2f%%')
plt.legend()
plt.show()


# In[30]:


sns.countplot(x=titanic['Sex'],hue=titanic['Survived']) # In Sex (0 represents female and 1 represents male)
plt.show()


# In[31]:


sns.countplot(x=titanic['Embarked'],hue=titanic['Sex'])
plt.show()


# In[32]:


sns.countplot(x=titanic['Pclass'])
plt.show()


# In[33]:


sns.countplot(x=titanic['Pclass'],hue=titanic['Sex'])
plt.show()


# In[34]:


sns.kdeplot(x=titanic['Age'])
plt.show()


# In[35]:


print(titanic['Survived'].value_counts())
sns.countplot(x=titanic['Survived'])
plt.show()


# In[36]:


sns.countplot(x=titanic['Parch'],hue=titanic['Survived'])
plt.show()


# In[37]:


sns.countplot(x=titanic['SibSp'],hue=titanic['Survived'])
plt.show()


# In[38]:


sns.countplot(x=titanic['Embarked'],hue=titanic['Survived'])
plt.show()


# In[39]:


sns.kdeplot(x=titanic['Age'],hue=titanic['Survived'])
plt.show()


# In[40]:


titanic.hist(figsize=(10,10))
plt.show()


# In[41]:


sns.boxplot(titanic)
plt.show()


# In[ ]:





# In[42]:


titanic.corr()


# In[43]:


sns.heatmap(titanic.corr(),annot=True,cmap='coolwarm')
plt.show()


# In[44]:


sns.pairplot(titanic)
plt.show()


# In[45]:


titanic['Survived'].value_counts()


# In[46]:


sns.countplot(x=titanic['Survived'])
plt.show()


# In[47]:


from sklearn.preprocessing import LabelEncoder
# Create an instance of LabelEncoder
le = LabelEncoder()

# Apply label encoding to each categorical column
for column in ['Sex','Embarked']:
    titanic[column] = le.fit_transform(titanic[column])

titanic.head()


# In[48]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[49]:


cols = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
x = titanic[cols]
y = titanic['Survived']
print(x.shape)
print(y.shape)
print(type(x))  # DataFrame
print(type(y))  # Series


# In[50]:


x.head()


# In[51]:


y.head()


# In[52]:


print(891*0.10)


# In[53]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[54]:


def cls_eval(ytest,ypred):
    cm = confusion_matrix(ytest,ypred)
    print('Confusion Matrix\n',cm)
    print('Classification Report\n',classification_report(ytest,ypred))

def mscore(model):
    print('Training Score',model.score(x_train,y_train))  # Training Accuracy
    print('Testing Score',model.score(x_test,y_test))     # Testing Accuracy


# In[55]:


lr = LogisticRegression(max_iter=1000,solver='liblinear')
lr.fit(x_train,y_train)


# In[56]:


mscore(lr)


# In[57]:


ypred_lr = lr.predict(x_test)
print(ypred_lr)


# In[58]:


cls_eval(y_test,ypred_lr)
acc_lr = accuracy_score(y_test,ypred_lr)
print('Accuracy Score',acc_lr)


# In[59]:


knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train,y_train)


# In[60]:


mscore(knn)


# In[61]:


ypred_knn = knn.predict(x_test)
print(ypred_knn)


# In[62]:


cls_eval(y_test,ypred_knn)
acc_knn = accuracy_score(y_test,ypred_knn)
print('Accuracy Score',acc_knn)


# In[63]:


svc = SVC(C=1.0)
svc.fit(x_train, y_train)


# In[64]:


mscore(svc)


# In[65]:


ypred_svc = svc.predict(x_test)
print(ypred_svc)


# In[66]:


cls_eval(y_test,ypred_svc)
acc_svc = accuracy_score(y_test,ypred_svc)
print('Accuracy Score',acc_svc)


# In[67]:


rfc=RandomForestClassifier(n_estimators=80,criterion='entropy',min_samples_split=5,max_depth=10)
rfc.fit(x_train,y_train)


# In[68]:


mscore(rfc)


# In[69]:


ypred_rfc = rfc.predict(x_test)
print(ypred_rfc)


# In[70]:


cls_eval(y_test,ypred_rfc)
acc_rfc = accuracy_score(y_test,ypred_rfc)
print('Accuracy Score',acc_rfc)


# In[71]:


dt = DecisionTreeClassifier(max_depth=5,criterion='entropy',min_samples_split=10)
dt.fit(x_train, y_train)


# In[72]:


mscore(dt)


# In[73]:


ypred_dt = dt.predict(x_test)
print(ypred_dt)


# In[74]:


cls_eval(y_test,ypred_dt)
acc_dt = accuracy_score(y_test,ypred_dt)
print('Accuracy Score',acc_dt)


# In[75]:


ada_boost  = AdaBoostClassifier(n_estimators=80)
ada_boost.fit(x_train,y_train)


# In[76]:


mscore(ada_boost)


# In[77]:


ypred_ada_boost = ada_boost.predict(x_test)


# In[78]:


cls_eval(y_test,ypred_ada_boost)
acc_adab = accuracy_score(y_test,ypred_ada_boost)
print('Accuracy Score',acc_adab)


# In[79]:


models = pd.DataFrame({
    'Model': ['Logistic Regression','knn','SVC','Random Forest Classifier','Decision Tree Classifier','Ada Boost Classifier'],
    'Score': [acc_lr,acc_knn,acc_svc,acc_rfc,acc_dt,acc_adab]})

models.sort_values(by = 'Score', ascending = False)


# In[80]:


colors = ["blue", "green", "red", "yellow","orange","purple"]

sns.set_style("whitegrid")
plt.figure(figsize=(15,5))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=models['Model'],y=models['Score'], palette=colors )
plt.show()


# In[ ]:




