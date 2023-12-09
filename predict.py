# importing required libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier


# loading and reading the dataset

heart_data = pd.read_csv("heart_disease_data.csv")
print("original data size ",heart_data.shape)
print(heart_data.isnull().sum())
# creating a copy of dataset so that will not affect our original dataset.
heart_d = heart_data.copy()

###############################    EM   #########################################
from sklearn.mixture import GaussianMixture
# Identify the columns with missing values
missing_cols = heart_d.columns[heart_d.isnull().any()]

# Loop over the columns with missing values
for col in missing_cols:
    # Separate the observed and missing values
    observed = heart_d[col][heart_d[col].notnull()]
    missing = heart_d[col][heart_d[col].isnull()]

    # Fit a GMM to the observed values
    gmm = GaussianMixture(n_components=2, random_state=0) # You can change the number of components as needed
    gmm.fit(observed.values.reshape(-1, 1))

    # Sample from the GMM to impute the missing values
    imputed = gmm.sample(missing.shape[0])[0].flatten()

    # Replace the missing values with the imputed values
    heart_d[col][heart_d[col].isnull()] = imputed

# Save the imputed dataset
heart_d.to_csv('imputed_heart_dataset.csv', index=False)

#___________________________________________________________________

# model building 
# loading and reading the dataset
imputed_data=pd.read_csv('imputed_heart_dataset.csv')
print("imputed data size: ",imputed_data.shape)
print(imputed_data.isnull().sum())

# creating a copy of dataset so that will not affect our original dataset.
heart_df = imputed_data.copy()

#fixing our data in x and y. Here y contains target data and X contains rest all the features.
x= heart_df.drop(columns= 'target')
y= heart_df.target

# splitting our dataset into training and testing for this we will use train_test_split library.
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=42)

#feature scaling
scaler= StandardScaler()
x_train_scaler= scaler.fit_transform(x_train)
x_test_scaler= scaler.fit_transform(x_test)

#_________________________________________________________

# creating K-Nearest-Neighbor classifier
model=RandomForestClassifier(n_estimators=20)
model.fit(x_train_scaler, y_train)
y_pred= model.predict(x_test_scaler)
p = model.score(x_test_scaler,y_test)
print(p)

print('Classification Report\n', classification_report(y_test, y_pred))
print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Creating a pickle file for the classifier
filename = 'heart-disease-prediction-knn-model.pkl'
pickle.dump(model, open(filename, 'wb'))

#___________________________________________________________________

# creating Logistic Regression Model
LR_model= LogisticRegression()
LR_model.fit(x_train_scaler, y_train)
y_pred_LR= LR_model.predict(x_test_scaler)
LR_model.score(x_test_scaler,y_test)
print('Classification Report\n', classification_report(y_test, y_pred_LR))
print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred_LR)*100),2)))

cm = confusion_matrix(y_test, y_pred_LR)
print(cm)

#_____________________________________________________________________
# creating Knn Model
Knn_model= KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
Knn_model.fit(x_train_scaler, y_train)
y_pred_knn= Knn_model.predict(x_test_scaler)
Knn_model.score(x_test_scaler,y_test)

print('Classification Report\n', classification_report(y_test, y_pred_knn))
print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred_knn)*100),2)))

cm = confusion_matrix(y_test, y_pred_knn)
print(cm)

#________________________________________________________________________

# Creating a pickle file for the classifier
filename = 'heart-disease-prediction-logistic.pkl'
pickle.dump(LR_model, open(filename, 'wb'))

