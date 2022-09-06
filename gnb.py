#import

import pandas as pd
import numpy as np

#iris = pd.read_csv("gnb_normalize_dataset_fix.csv")
iris = pd.read_csv("dataset_enzim_5_pca.csv")
#iris = pd.read_csv("dataset_enzim_3_pca.csv")
#iris = pd.read_csv("dataset_enzim_selected_feature.csv")


iris.head()

#  variabel bebas
x = iris.drop(["class"], axis = 1)
x.head()

#variabel tidak bebas
y = iris["class"]
y.head()

# classification 
# please install scikit library 
# pip install -U scikit-learn

# separate the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 3)
#import from library 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

# Call Gaussian Naive Bayes 
iris_model = GaussianNB()




# Insert the training dataset to  Naive Bayes function
NB_train = iris_model.fit(x_train, y_train)

# Next step: Prediction the x_test to the model built and save to the             y_pred variable 
# show the result of prediction 
y_pred = NB_train.predict(x_test)
np.array(y_pred) 

# show the y_test based on separation dataset
np.array(y_test)

# show the confusion matrix based on the prediction result 
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


#evaluate performance from the confusion matrix 
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))

# this value will show all probability for each predicted class 
NB_train.predict_proba(x_test)





