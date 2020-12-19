#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas  as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


data = pd.read_csv('/cxldata/projects/creditcard.csv')

data.head(10)

data.shape


# In[10]:


data.describe()
data.isnull().sum()


# In[11]:


X = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']

print(data['Class'].value_counts())

print('Valid Transactions: ', round(data['Class'].value_counts()[0]/len(data) * 100,2), '% of the dataset')
print('Fraudulent Transactions: ', round(data['Class'].value_counts()[1]/len(data) * 100,2), '% of the dataset')


# In[15]:


colors = ['blue','red']

sns.countplot('Class', data=data, palette=colors)


# In[16]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Transactions in X_train dataset: ", X_train.shape)
print("Transaction classes in y_train dataset: ", y_train.shape)

print("Transactions in X_test dataset: ", X_test.shape)
print("Transaction classes in y_test dataset: ", y_test.shape)


# In[22]:


from sklearn.preprocessing import StandardScaler

scaler_amount = StandardScaler()
scaler_time = StandardScaler()

X_train['normAmount'] = scaler_amount.fit_transform(X_train['Amount'].values.reshape(-1, 1))
X_test['normAmount'] = scaler_amount.transform(X_test['Amount'].values.reshape(-1, 1))

X_train['normTime'] = scaler_time.fit_transform(X_train['Time'].values.reshape(-1, 1))
X_test['normTime'] = scaler_time.transform(X_test['Time'].values.reshape(-1, 1))

X_train = X_train.drop(['Time', 'Amount'], axis=1)
X_test = X_test.drop(['Time', 'Amount'], axis=1)

X_train.head()


# In[30]:


from imblearn.over_sampling import SMOTE

print("Before over-sampling\n:", y_train['Class'].value_counts())

sm = SMOTE()

X_train_res, y_train_res = sm.fit_sample(X_train, y_train['Class'])

print("After over-sampling:\n", y_train_res.value_counts())


# In[35]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, auc, roc_curve

parameters = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

lr = LogisticRegression()

clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)

k = clf.fit(X_train_res, y_train_res)

print(k.best_params_)


# In[44]:


lr_gridcv_best = clf.best_estimator_

y_test_pre = lr_gridcv_best.predict(X_test)

cnf_matrix_test = confusion_matrix(y_test, y_test_pre)

print("Recall metric in the test dataset:", (cnf_matrix_test[1,1]/(cnf_matrix_test[1,0]+cnf_matrix_test[1,1] )))


y_train_pre = lr_gridcv_best.predict(X_train_res)

cnf_matrix_train = confusion_matrix(y_train_res, y_train_pre)

print("Recall metric in the train dataset:", (cnf_matrix_train[1,1]/(cnf_matrix_train[1,0]+cnf_matrix_train[1,1] )))


# In[50]:


from sklearn.metrics import plot_confusion_matrix

class_names = ['Not Fraud', 'Fraud']

plot_confusion_matrix(k, X_test, y_test,  values_format = '.5g', display_labels=class_names) 
plt.title("Test data Confusion Matrix")
plt.show()

plot_confusion_matrix(k, X_train_res, y_train_res,  values_format = '.5g', display_labels=class_names)
plt.title("Oversampled Train data Confusion Matrix")
plt.show()


# In[53]:


y_k =  k.decision_function(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_k)

roc_auc = auc(fpr,tpr)

print("ROC-AUC:", roc_auc)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




