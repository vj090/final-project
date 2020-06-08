# -*- coding: utf-8 -*-
"""
Created on Tue 2 14:55:19 2019

@author: vinod
"""

"""
#############################################################################
DataSet - Predicting Diabetes(pima-indians-diabetes)
#############################################################################

"""

#Import Libraries
#-------------------------------
import pandas as pd #pandas is a dataframe library
import matplotlib.pyplot as plt #matplotlib.pyplot plots data
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from sklearn import preprocessing
import numpy as np
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn import tree
from sklearn.externals.six import StringIO 
import pydotplus
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import statsmodels.api as sm


#Load and Review data
#-----------------------------
df = pd.read_csv("pima-data.csv") # load Pima data.

df.shape

df.head(5)

df.tail(5)

#Stats Analysis with Standard Deviation and Mean
#------------------------------------------------------
desc = df.describe()
desc


############################ Visualization  ###############################

#Box-Plot(Eda)
#------------------
df.boxplot(column='glucose_conc')
df.boxplot(column='skin')

#Scatter-Plot
#------------------
plt.scatter(df.age, df.skin, edgecolors='r')
plt.scatter(df.bmi, df.num_preg, edgecolors='r')
plt.scatter(df.diab_pred, df.diastolic_bp, edgecolors='r')

#Histogram
#-----------------
df.age.hist()
df.bmi.hist()
df.glucose_conc.hist()


#Corelation between variables
#---------------------------------
corr = df.corr()
corr

#Visual Graph of Corelation
#----------------------------------
sns.heatmap(corr, annot = True)


def plot_corr(df, size=11):
    corr=df.corr() #data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr) #colour code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns) # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns) #draw y tick marks
    
plot_corr(df)
    
df.corr()


#####################  Exploratory Data Analysis (EDA) ######################

#Check for null values
#-------------------------------------
cols = list(df.columns)
print(cols)

for c in cols:
    if (len(df[c][df[c].isnull()])) > 0:
        print("WARNING: Column '{}' has NULL values".format(c))
    else:
        print("No Null values present in %s" %c)
        
    
# Delete the skin variable as it ios not use for the prediction
#----------------------------------------------------------------    
del df['skin']
    
#Check for theData Types
#--------------------------------------    
df.head(5)
    
diabetes_map = {True : 1, False : 0}
    
df['diabetes'] = df['diabetes'].map(diabetes_map)
    
df.head()
    
#Check true/false ratio
#--------------------------------------  
num_true = len(df.loc[df['diabetes'] == True])
num_false = len(df.loc[df['diabetes'] == False])
print("Number of True cases: {0} ({1:2.2f}%)".format(num_true, (num_true/(num_true+num_false))*100))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, (num_false/(num_true+num_false))*100))
    

############################ Training And Testing ###############################

#Splitting the data in the ratio of 70% and 30%
#----------------------------------------------------
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split

feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

X = df[feature_col_names].values # predictor feature columns (8 X m)
y = df[predicted_class_names].values # predicted class (1=true, 0=false) column (1 X m)
split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42)

print("{0:0.2f}% in training set".format((len(X_train)/len(df.index)) * 100))
print("{0:0.2f}% in test set".format((len(X_test)/len(df.index)) * 100))


#Verifying predicted value
#--------------------------------
print("Original True: {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes'] == 1]), (len(df.loc[df['diabetes'] == 1])/len(df.index)) * 100))
print("Original False: {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes'] == 0]), (len(df.loc[df['diabetes'] == 0])/len(df.index)) * 100))

print("Training True: {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train)) * 100))
print("Training False: {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train)) * 100))

print("Test True: {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test)) * 100))
print("Test True: {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test)) * 100))


df.head()

#Impute with the mean
#-----------------------
from sklearn.preprocessing import Imputer

fill_0 = Imputer(missing_values=0, strategy="mean", axis=0)

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)


############################## ALGORITHMS  #################################


#1) Naive Bayes Algorithm(GaussianNB)
#-----------------------------------

from sklearn.naive_bayes import GaussianNB

# Create Gaussian Naive Bayes model object and train it with the data
nb_model = GaussianNB()

nb_model.fit(X_train, y_train.ravel())


#Performance on Training Data
#------------------------------------------
# predict values using the training data
nb_predict_train = nb_model.predict(X_train)
nb_predict_train

# import the performance metrics library
from sklearn import metrics

# Accuracy
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, nb_predict_train)))


# predict values using the testing data
nb_predict_test = nb_model.predict(X_test)

from sklearn import metrics

# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))

#Confusion Metrics:
#---------------------

print("Confusion Matrix")
# Note the use of labels for set 1=True to upper left and 0=False to lower left
print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test, labels=[1, 0])))


print("Classification Report")
print(metrics.classification_report(y_test, nb_predict_test, labels=[1, 0]))




#2)Random Forest:
#------------------
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state = 42) # Create random forest object
rf_model.fit(X_train, y_train.ravel())

#Predict Training Data:
#-------------------------------
rf_predict_train = rf_model.predict(X_train)
# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, rf_predict_train)))


rf_predict_test = rf_model.predict(X_test)
rf_predict_test
# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, rf_predict_test)))

print("{0}".format(metrics.confusion_matrix(y_test, rf_predict_test, labels=[1, 0])))


print("Classification Report")
print(metrics.classification_report(y_test, rf_predict_test, labels=[1, 0]))





#3)Logistic Regression:
#-----------------------------------

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(C=0.7, random_state=42)
lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
print(metrics.confusion_matrix(y_test, lr_predict_test, labels=[1, 0]))

print("Classification Report")
print(metrics.classification_report(y_test, lr_predict_test, labels=[1, 0]))

#LogisticRegressionCV
#----------------------------
from sklearn.linear_model import LogisticRegressionCV
lr_cv_model = LogisticRegressionCV(n_jobs=-1, random_state=42, Cs=3, cv=10, refit=True, class_weight="balanced")
lr_cv_model.fit(X_train, y_train.ravel())



#Predict on Test data
#----------------------------
lr_cv_predict_test = lr_cv_model.predict(X_test)

# Training metrics
#----------------------------
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_cv_predict_test)))
print(metrics.confusion_matrix(y_test, lr_cv_predict_test, labels=[1, 0]))
print("Classification Report")
print(metrics.classification_report(y_test, lr_cv_predict_test, labels=[1, 0]))





#4)K-Nearest Neighbour:
#-----------------------------

# standardize the dataset
# ------------------>
glass_scaled = df.copy(deep=True)
minmax = preprocessing.MinMaxScaler()
scaledvals = minmax.fit_transform(glass_scaled.iloc[:,0:8])
glass_scaled.loc[:,0:8] = scaledvals
glass_scaled.head(10)

df.head(10)


# split the scaled dataset into X and Y variables. 
# These are numpy array
# --------------------------------------------------------------------->
X=glass_scaled.values[:,0:8]
Y=glass_scaled.values[:,8]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42)

X_train.shape
X_test.shape
y_train.shape
y_test.shape


# cross-validation to select optimum neighbours and folds
# -------------------------------------------------------

# creating a list of nearest neighbours
# -------------------------------------
# nn=list(range(3,50,2))
nn=list(range(3,10,1))
print(nn)
len(nn)

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
# ---------------------------------
for k in nn:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, 
                             scoring='accuracy')
    scores=np.around(scores.astype(np.double),4)
    
    cv_scores.append(scores.mean())
print(cv_scores)

MSE = [1 - x for x in cv_scores]
MSE
# determining best k
optimal_k = nn[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)

# knn algorithm call
# ------------------>
clf = neighbors.KNeighborsClassifier(n_neighbors=optimal_k)
fit1 = clf.fit(X_train, y_train)
print(fit1)

# predict
# ------------------>
y_pred = fit1.predict(X_test)
y_pred
# prediction report
# ----------------->
print(metrics.classification_report(
        y_test, y_pred))



#5)Decision Tree
#--------------------------

# gini model
# entropy model
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42)

X_train.shape
X_test.shape
y_train.shape
y_test.shape


clf_gini = dtc(criterion = "gini", random_state = 100, 
               max_depth=3, min_samples_leaf=5)

fit1 = clf_gini.fit(X_train, y_train)
print(fit1)


# tree visualisation
# -------------------------------------
dot_data = StringIO()

tree.export_graphviz(fit1, out_file=dot_data,  
                filled=True, rounded=True, special_characters=True)

graph=pydotplus.graph_from_dot_data(dot_data.getvalue())


# predictions
# -------------------------------------
pred_gini = fit1.predict(X_test)
pred_gini
len(y_test)
len(pred_gini)
print("Accuracy is ", accuracy_score(y_test,pred_gini)*100)


# confusion matrix of the Y-variables for both models
# -----------------------------------------------------------
confusion_matrix(y_test, pred_gini)

           


    
    
    





    
