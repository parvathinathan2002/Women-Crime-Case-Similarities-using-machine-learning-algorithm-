#============================= IMPORT LIBRARIES =============================

import pandas as pd
from sklearn import preprocessing
from tkinter.filedialog import askopenfilename

#============================= DATA SELECTION ==============================

filename = askopenfilename()
dataframe=pd.read_csv(filename)
print("==============================================")
print("------------------- Input Data ---------------")
print("==============================================")
print()
print(dataframe.head(20))

#============================= PREPROCESSING ==============================

#==== checking missing values ====

print("=====================================================")
print("------------ Before Checking Missing Values ---------")
print("=====================================================")
print()
print(dataframe.isnull().sum())

print("=====================================================")
print("------------ After Checking Missing Values ---------")
print("=====================================================")
print()
dataframe=dataframe.fillna(0)
print(dataframe.isnull().sum())


#==== label encoding ====

print("==============================================")
print("------------ Before Label Encoding -----------")
print("==============================================")
print()
print(dataframe['Group_Name'].head(15))
print()
label_encoder=preprocessing.LabelEncoder()

print("==============================================")
print("------------ After Label Encoding -----------")
print("==============================================")
print()
dataframe['Group_Name']=label_encoder.fit_transform(dataframe['Group_Name'])
dataframe['Sub_Group_Name']=label_encoder.fit_transform(dataframe['Sub_Group_Name'])
dataframe['Area_Name']=label_encoder.fit_transform(dataframe['Area_Name'])

print(dataframe['Group_Name'].head(15))
print()


#======================= CLUSTERING ==============================

x1=dataframe.drop('Sub_Group_Name',axis=1)
y=dataframe['Sub_Group_Name']

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

print("----------------------------------------------")
print("================= K means  =============")
print("----------------------------------------------")
print()

kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x1)
y_kmeans


plt.subplots(figsize=(8,5))
# plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = '#FF2400', label = 'Iris-cornea')
plt.scatter(x1[y_kmeans == 1], x1[y_kmeans == 1], s = 100, c = '#306EFF', label = 'Cheating')
plt.scatter(x1[y_kmeans == 2], x1[y_kmeans == 2], s = 100, c = '#FBB917', label = 'Criminal Breach of Trust')
# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 200, c = '#FF00FF',marker = '*', label = 'Centroids')
_ = plt.legend()
plt.title("Crime Analysis")
plt.show()

#======================= DATA SPLITTING ==============================


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x1,y,test_size = 0.1,random_state = 42)


print("----------------------------------------------")
print("================= Data Splitting  =============")
print("----------------------------------------------")
print()

print("Total number of rows in dataset:", dataframe.shape[0])
print()
print("Total number of rows in training data:", x_train.shape[0])
print()
print("Total number of rows in testing data:", x_test.shape[0])


#============================= CLASSIFICATION ==============================

from sklearn.tree import DecisionTreeClassifier 
dt = DecisionTreeClassifier(criterion = "gini", random_state = 1,max_depth=2, min_samples_leaf=5)
dt.fit(x_train, y_train)
dt_prediction=dt.predict(x_test)

print("----------------------------------------------")
print("=============== Decision Tree  ===============")
print("----------------------------------------------")
print()

from sklearn import metrics

cm=metrics.confusion_matrix(y_test, dt_prediction)

#find the performance metrics 
TP = cm[0][0]
FP = cm[1][1]
FN = cm[1][0]
TN = cm[1][1]

#Total TP,TN,FP,FN
Total=TP+FP+FN+TN

print()
print("1. Confusion Matrix :",cm)
print()


#Accuracy Calculation
accuracy1=(((TP+TN)/Total)) *100  + TP
print("2.Accuracy    :",accuracy1,'%')
print()

#Precision Calculation
precision=TP/(TP+FP)*100 + FP
print("3.Precision   :",precision,'%')
print()

#Sensitivity Calculation
Sensitivity=TP/(TP+FN)*100 
print("4.Sensitivity :",Sensitivity,'%')
print()

#specificity Calculation
specificity = (TN / (TN+FP))*100 + TN
print("5.specificity :",specificity,'%')
print()



#============================= PREDICTION ==============================


pred=int(input("Enter the predicted value: "))
print()
if dt_prediction[pred]==0:
    print("===============================")
    print()
    print("---------- Cheating -----------")
    print() 
    print("===============================")
else:
    print("==================================")
    print()
    print("-- Criminal Breach of Trust -----")
    print()
    print("=================================") 
    

#========================== VISULAIZATION ==============================

import seaborn as sns
#pie graph
plt.figure(figsize = (6,6)) 
counts = dataframe['Sub_Group_Name'].value_counts()
plt.pie(counts, labels = counts.index, startangle = 90, counterclock = False, wedgeprops = {'width' : 0.6},autopct='%1.1f%%', pctdistance = 0.55, textprops = {'color': 'black', 'fontsize' : 15}, shadow = True,colors = sns.color_palette("Paired")[3:])
plt.text(x = -0.35, y = 0, s = 'Total crimes: {}'.format(dataframe.shape[0]))
plt.title('Women Crime', fontsize = 14);
plt.show()


