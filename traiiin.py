import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import opendatasets as od
#od.download('https://www.kaggle.com/datasets/prudhvignv/milk-grading')
data=pd.read_csv('Milk Grading (1).csv')
data.head()
data
data.shape
data.info()
print(data.isnull().sum())
print(data.describe())
print(data.dtypes)
print(data['Fat '].value_counts())
print(data['Grade'].value_counts())
# Ensure 'Grade' column is converted to string type
# Ensure 'Grade' column is a string type and handle the conversion properly
data['Grade'] = data['Grade'].astype(str)

# Map the 'Grade' column to 'Bad', 'Moderate', 'Good' based on the value
data.loc[data['Grade'] == '0.0', 'Grade'] = 'Bad'
data.loc[data['Grade'] == '0.5', 'Grade'] = 'Moderate'
data.loc[data['Grade'] == '1.0', 'Grade'] = 'Good'

# Create a new 'Grade_Numeric' column that maps 'Bad' -> 0, 'Moderate' -> 0.5, 'Good' -> 1
data['Grade_Numeric'] = data['Grade'].map({'Bad': 0.0, 'Moderate': 0.5, 'Good': 1.0})

# Now, select only the numeric columns to calculate correlation
numeric_data = data.select_dtypes(include=['number'])

# Print the correlation matrix for only numeric columns
print(numeric_data.corr())

# If you still want to include the entire dataset, use:
# print(data.corr())

import matplotlib.pyplot as plt
import seaborn as sns

# Exclude non-numeric columns from the correlation calculation
numeric_data = data.select_dtypes(include=['number'])

# Plot the heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
data.hist(bins=50, figsize=(18,18))
sns.displot(data['Temprature'])
plt.title('Colour')
plt.show()
def countplot_of_2(x,hue,title=None,figsize=(6,5)):
    plt.figure(figsize=figsize)
    sns.countplot(data=data[[x,hue]],x=x,hue=hue)
    plt.title(title)
    plt.show()
countplot_of_2('Grade','Temprature','Good/Bad/Moderate Vs 40/55')
import seaborn as sns
sns.stripplot(x="Temprature", y="Grade", hue="Taste", data=data)
Grade = ['Bad', 'Moderate', 'Good']
 
cate = [20, 15, 30]
 
# Creating plot
fig = plt.figure(figsize =(11, 7))
plt.pie(cate, labels = Grade)
 
# show plot
plt.show()
sns.barplot(x = 'Grade',
            y = 'pH',
            data = data)
 
# Show the plot
plt.show()
# create boxplot Colour Vs Grade
bplot = sns.boxplot(y='Colour', hue='Grade', 
                 data=data, 
                 width=0.5,
                 palette="colorblind")
plt.show()

sns.scatterplot(data = data, x = "Temprature", y = "Grade")

plt.show()
sns.scatterplot(data = data, x = "Odor", y = "Grade")

plt.show()
sns.pairplot(data)
plt.show()
sns.boxplot(data['pH'])
plt.show()
data.drop(columns=['Grade_Numeric'], inplace=True)
y = data['Grade']
x = data.drop(columns=['Grade'],axis=1)
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)
from imblearn import over_sampling
os = over_sampling.RandomOverSampler(random_state=0)
os
x,y = os.fit_resample(x,y)
print(y.value_counts())
print(x.head())
print(y.tail())
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svc= SVC()
svc.fit(x_train,y_train)
y_predict=svc.predict(x_test)
test_accuracy=accuracy_score(y_test,y_predict)
print(test_accuracy)
y_train_predict=svc.predict(x_train)
train_accuracy=accuracy_score(y_train,y_train_predict)
print(train_accuracy)
print(pd.crosstab(y_test,y_predict))
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_predict,zero_division=0))
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)

y_predict1=rfc.predict(x_test)
test_accuracy=accuracy_score(y_test,y_predict1)
print(test_accuracy)
y_train_predict1=rfc.predict(x_train)
train_accuracy=accuracy_score(y_train,y_train_predict1)
print(train_accuracy)
print(pd.crosstab(y_test,y_predict1))
print(classification_report(y_test,y_predict1))
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(x_train, y_train)

y_predict2=dtc.predict(x_test)
test_accuracy=accuracy_score(y_test,y_predict2)
print(test_accuracy)
y_train_predict2=dtc.predict(x_train)
train_accuracy=accuracy_score(y_train,y_train_predict2)
print(train_accuracy)
print(y)
print(pd.crosstab(y_test,y_predict2))
print(classification_report(y_test,y_predict2))
from sklearn.ensemble import ExtraTreesClassifier
etc=ExtraTreesClassifier()
etc.fit(x_train,y_train)

y_predict3=etc.predict(x_test)
test_accuracy=accuracy_score(y_test,y_predict3)
print(test_accuracy)
y_train_predict3=etc.predict(x_train)
train_accuracy=accuracy_score(y_train,y_train_predict3)
print(train_accuracy)
print(pd.crosstab(y_test,y_predict3))
print(classification_report(y_test,y_predict3))
from sklearn.model_selection import GridSearchCV
parameters = {
              "kernel":['linear', 'rbf', 'sigmoid'],"gamma":['scale', 'auto'],
              "break_ties":[True,False]   
              }
from sklearn.model_selection import KFold
svc=SVC()
gdcv = GridSearchCV(estimator=svc,param_grid=parameters)
gdcv.fit(x_train,y_train)
gdcv.best_params_
from sklearn.metrics import accuracy_score
svc=SVC(kernel='rbf',gamma='auto',break_ties=True)
svc.fit(x_train,y_train)
y_train_pred=svc.predict(x_train)
y_test_pred=svc.predict(x_test)
print("train accuracy",accuracy_score(y_train_pred,y_train))
print("test accuracy",accuracy_score(y_test_pred,y_test))
parameters={"n_estimators" : [2,5,10,15,20,25],
            "warm_start":[False],"min_samples_split":[2],"criterion":['entropy'],"random_state":[111]
    }
rfc=RandomForestClassifier(warm_start=False)
gdcv1 = GridSearchCV(estimator=rfc,param_grid=parameters)
gdcv1.fit(x_train,y_train)
gdcv1.best_params_
from sklearn.metrics import accuracy_score
rfc=RandomForestClassifier(criterion='entropy',min_samples_split=2,n_estimators=5,warm_start=False,random_state=111)
rfc.fit(x_train,y_train)
y_train_pred=rfc.predict(x_train)
y_test_pred=rfc.predict(x_test)
print("train accuracy",accuracy_score(y_train_pred,y_train))
print("test accuracy",accuracy_score(y_test_pred,y_test))
parameters={"n_estimators":[2,5,10,15,20,25],"criterion":['entropy'],
            "min_samples_split":[2],
             "min_samples_leaf":[1],"random_state":[111]}
etc=ExtraTreesClassifier()
gdcv2 = GridSearchCV(estimator=etc,param_grid=parameters)
gdcv2.fit(x_train,y_train)
gdcv2.best_params_
from sklearn.metrics import accuracy_score
etc=ExtraTreesClassifier(min_samples_leaf=1,min_samples_split=2,n_estimators=5,criterion='entropy',random_state= 111)
etc.fit(x_train,y_train)
y_train_pred=etc.predict(x_train)
y_test_pred=etc.predict(x_test)
print("train accuracy",accuracy_score(y_train_pred,y_train))
print("test accuracy",accuracy_score(y_test_pred,y_test))
parameters={"criterion":['entropy'],
    "splitter":['best'],
    "min_samples_split":[2],"random_state":[111]}
dtc=DecisionTreeClassifier()
gdcv3 = GridSearchCV(estimator=dtc,param_grid=parameters)
gdcv3.fit(x_train,y_train)
gdcv3.best_params_
from sklearn.metrics import accuracy_score
dtc=DecisionTreeClassifier(splitter='best',min_samples_split=2,criterion='entropy', random_state=111)
dtc.fit(x_train,y_train)
y_train_pred=dtc.predict(x_train)
y_test_pred=dtc.predict(x_test)
print("train accuracy",accuracy_score(y_train_pred,y_train))
print("test accuracy",accuracy_score(y_test_pred,y_test))
import pickle
pickle.dump(svc,open('milkgrade1.pkl','wb'))
print(x_train.shape)  # Check how many features were used for training
print(x.shape)  # Print the shape of the input data you're passing to the model
