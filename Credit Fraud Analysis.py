#!/usr/bin/env python
# coding: utf-8

# In[113]:


# Import the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Import the data
Base = pd.read_csv("Base.csv")

#print(Base.info())


# In[114]:


# Check for missing values
#print(Base.isnull().sum())


# In[115]:




# Group data by customer age and fraud bool
age_fraud = Base.groupby(['customer_age', 'fraud_bool']).size().unstack(fill_value=0)

# Create stacked bar chart
ax = age_fraud.plot(kind='bar', stacked=True, figsize=(10, 6))

# Set chart title and labels
ax.set_title('Fraud by Customer Age')
ax.set_xlabel('Customer Age')
ax.set_ylabel('Number of Applications')

# Display chart
plt.show()


# In[116]:




# Group data by customer icome and fraud bool
income_fraud = Base.groupby(['income', 'fraud_bool']).size().unstack(fill_value=0)

# Create stacked bar chart
ax = income_fraud.plot(kind='bar', stacked=True, figsize=(10, 6))

# Set chart title and labels
ax.set_title('income')
ax.set_xlabel('income')
ax.set_ylabel('Number of Applications')

# Display chart
plt.show()


# In[117]:




# Group data by customer employement status and fraud bool
employment_status_fraud = Base.groupby(['employment_status', 'fraud_bool']).size().unstack(fill_value=0)

# Create stacked bar chart
ax = employment_status_fraud.plot(kind='bar', stacked=True, figsize=(10, 6))

# Set chart title and labels
ax.set_title('employment_status')
ax.set_xlabel('employment_status')
ax.set_ylabel('Number of Applications')

# Display chart
plt.show()


# In[118]:




# Group data by days since request and fraud bool
days_since_request_fraud = Base.groupby(['days_since_request', 'fraud_bool']).size().unstack(fill_value=0)

# Create stacked bar chart
ax = employment_status_fraud.plot(kind='bar', stacked=True, figsize=(10, 6))

# Set chart title and labels
ax.set_title('days_since_request')
ax.set_xlabel('days_since_request')
ax.set_ylabel('Number of Applications')

# Display chart
plt.show()


# In[119]:


# Convert categorical variables to numeric
Base = pd.get_dummies(Base, columns=['payment_type', 'employment_status', 'housing_status'])


# In[120]:


# Standardize numerical variables
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
num_vars = ['income', 'name_email_similarity', 'customer_age', 'days_since_request', 'intended_balcon_amount', 'proposed_credit_limit']
Base[num_vars] = scaler.fit_transform(Base[num_vars])


# In[121]:


# Create a correlation matrix
corr_matrix = Base.corr().abs()

# Select upper triangle of correlation matrix
#upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
sns.set(font_scale=1.2)
plt.figure(figsize=(12, 8))
sns.heatmap(upper, cmap='YlGnBu', annot=True, fmt='.2f', annot_kws={"fontsize":10}, cbar_kws={'label': 'Correlation Coefficient'})
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
Base = Base.drop(to_drop, axis=1)


# In[122]:


# Drop the unnecessary features from the dataset
irrelevant_features = ['name_email_similarity', 'prev_address_months_count', 'source', 'zip_count_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 'email_is_free', 'phone_home_valid', 'device_os']


# In[123]:


Base.drop(irrelevant_features, axis=1, inplace=True)

# Print the remaining features in the dataset
print(Base.columns)


# In[124]:


print(Base.info())


# In[125]:


##LogisticRegression

#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#from sklearn.preprocessing import StandardScaler


# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Base.drop('fraud_bool', axis=1), Base['fraud_bool'], test_size=0.2, random_state=42)
scaler = StandardScaler()

# Fit scaler on training set
scaler.fit(X_train)

# Scale training set
X_train_scaled = scaler.transform(X_train)

# Scale testing set
X_test_scaled = scaler.transform(X_test)
# create logistic regression model
model = LogisticRegression()



# train the model on the training set
model.fit(X_train_scaled, y_train)

# make predictions on the testing set
y_pred = model.predict(X_test_scaled)

# evaluate model performance
accuracyL = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
# plot a histogram of the Class variable to see the distribution of fraud and non-fraud transactions
sns.countplot(x='fraud_bool', data=Base)

print('Accuracy:', accuracyL)
print('Confusion Matrix:', confusion_mat)
# evaluate the performance of the model
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)
sns.heatmap(confusion_mat, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[126]:


#Classification Report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred , target_names = ['0', '1'])) 


# In[127]:


from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize = (8, 8))

ax.plot(ax.get_xlim(), ax.get_ylim(), ls = '--', c = 'k')
RocCurveDisplay.from_estimator(model, 
                               X = X_test, 
                               y = y_test, 
                               ax = ax);


# In[128]:


##RandomForestClassifier

# Import the necessary libraries

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# Create a random forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the training data
rf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf.predict(X_test)

# Evaluate the accuracy of the model
accuracyR = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracyR)
# evaluate the performance of the model
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)


# In[129]:


##Naive Bayes model
##import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
##from sklearn.model_selection import train_test_split
##from sklearn.preprocessing import StandardScaler

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Base.drop('fraud_bool', axis=1), Base['fraud_bool'], test_size=0.2, random_state=42)

# scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# create Naive Bayes model
model = GaussianNB()

# train the model on the training set
model.fit(X_train_scaled, y_train)

# make predictions on the testing set
y_pred = model.predict(X_test_scaled)

# evaluate model performance
accuracyN = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

# print evaluation metrics
print('Accuracy:', accuracyN)
print('Confusion Matrix:', confusion_mat)
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_mat, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[130]:


##Evaluate the best model on the test set
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  GridSearchCV

dtc = DecisionTreeClassifier(random_state=42)
param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}

# Perform a grid search to find the best hyperparameters
grid_search = GridSearchCV(dtc, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_dtc = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_dtc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Best accuracy:", grid_search.best_score_)
print("Test accuracy:", accuracy)


# In[131]:


##finding best accuracy of three models
if accuracyL > accuracyR:
    if accuracyL> accuracyN:
        print ("Logistic Regression has better accuracy rate and the accuracy is: ", accuracyL)
    else:
        print (" Naive Bayes has better accuracy rate and the accuracy is: ", accuracyN)
else:
    if accuracyR > accuracyN:
        print ("Random forest has better accuracy rate and the accuracy is: ", accuracyR)
    else:
        print ("Naive Bayeshas better accuracy rate and the accuracy is: ", accuracyN)


# In[ ]:




