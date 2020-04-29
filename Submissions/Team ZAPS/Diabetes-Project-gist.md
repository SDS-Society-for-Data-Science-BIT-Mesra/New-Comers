># PROJECT:- Prediction of diabetes in Indian patients:
Project Members :Zeeshan Equbal , Abhraneil Bhattacharya, Shivam Kumar Tiwary , Pratyay Das  
Project Description : Our project was to deploy ML algorithms such as Naive Bayes classifer and Logistic Regression to 
predict wether a person has diabetes or not.

The summary of our project is as follows :

1. Importing necessary modules 
2. Collecting and Uploading our dataset
3. Data Cleaning
4  Simple Visualisation Techniques 
5. Choosing wether regression or classification
6. Data splitting and model selection.


>### IMPORTING NECESSARY MODULES :-


```import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```

</br>

>### DATASET COLLECTION AND UPLOAD :
```
We used the pima-indian diabetes from kaggle 
link:-  https://www.kaggle.com/uciml/pima-indians-diabetes-database/kernels
```
</br>


>### DATA CLEANING:--</br>
Initially our dataset had no defined columns , just raw data:
based on the kaggle websited we identified the columns and renamed them with the help of Pandas</br>
```
df.rename({'6': 'Pregnancies', '148': 'Glucose','72':'BloodPressure','35':'SkinThickness','0':'Insulin','33.6':'BMI','0.627':'DiabetesPedigreeFunction','50':'Age','1':'Outcome'}, axis=1 , inplace = True)
df.head(10)
we further explored our data types with df.info()
df.info() 
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 767 entries, 0 to 766
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Pregnancies               767 non-null    int64  
 1   Glucose                   767 non-null    int64  
 2   BloodPressure             767 non-null    int64  
 3   SkinThickness             767 non-null    int64  
 4   Insulin                   767 non-null    int64  
 5   BMI                       767 non-null    float64
 6   DiabetesPedigreeFunction  767 non-null    float64
 7   Age                       767 non-null    int64  
 8   Outcome                   767 non-null    int64  
dtypes: float64(2), int64(7)
```

>### SIMPLE VISUALISATIONS USING SEABORN:-
</br>
we wanted to get some serious insights into our data so we chose to visualise them :)

</br>

```
fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(df.Age, bins = 20, ax=ax[0,0]) 
sns.distplot(df.Pregnancies, bins = 20, ax=ax[0,1]) 
sns.distplot(df.Glucose, bins = 20, ax=ax[1,0]) 
sns.distplot(df.BloodPressure, bins = 20, ax=ax[1,1]) 
sns.distplot(df.SkinThickness, bins = 20, ax=ax[2,0])
sns.distplot(df.Insulin, bins = 20, ax=ax[2,1])
sns.distplot(df.DiabetesPedigreeFunction, bins = 20, ax=ax[3,0]) 
sns.distplot(df.BMI, bins = 20, ax=ax[3,1])
```

</br>
</br>
WE ALSO CREATED A CORRELATION PLOT TO DETERMINE WHICH FEATURES WERE BETTER FOR OUR USE
</br>
NEXT THING WAS SOMETHING THAT THE TEAM ALREADY KNEW AND THAT IS : OUR PROBLEM WAS A CLASSIFICATION PROBLEM
</br>
</br>

>### MODEL SELECTION AND PREDICTION:-

</br>
FINALLY HEADING ON TO MODEL SELECTION , WE CHOSE NAIVE BAYES , LOGISTIC REGRESSION AND KNN
UNLIKE KNN AND LOGISTIC REGRESSION WHICH WAS IMPORTED FROM SCIKIT LEARN, OUR NAIVE BAYES ALGORITHM WAS CREATED FROM SCRATCH
</br>
</br>

HERE IS THE MAIN() FROM OUR NAIVE BAYES ALGORITHM

```
def main():
    trainingSet, testSet=splitDataset(df, 0.83)
    print('Split {0} rows into train = {1} and test = {2} rows'.format(len(df), len(trainingSet), len(testSet)))
    # Prepare model
    summaries=summarizeByClass(trainingSet)
    # Test Model
    predictions=getPredictions(summaries, testSet)
    accuracy=getAccuracy(testSet, predictions)
    print('Accuracy: {}%'.format(accuracy))
    
main()
```

ACCURACY OF NAIVE BAYES:-
</br>
```
Split 768 rows into train = 637 and test = 131 rows
Accuracy: 76.33587786259542%
```
</br>
</br>
NEXT WE APPLIED LOGISTIC REGRESSION :-
</br>

````
logisticRegr.fit(x_train, y_train)
y_pred = logisticRegr.predict(x_test)
check=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
check.head(10)
````
</br>
WE CHECKED OUR ACCURACY NEXT :
</br>

```
check=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
check.head(10)

Accuracy: 95.83333333333334 %
```
</br>
FINALLY WE USED KNN FROM SCIKIT LEARN 
</br>

```
from sklearn.neighbors import KNeighborsClassifier
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.03, random_state=0) //TT RATIO 97:3
y_pred=neigh.predict(x_test)
check=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
check.head(10)
```

LASTLY WE CHECKED THE ACCURACY OF OUR LAST ALGORITHM

```
score = logisticRegr.score(x_test, y_test)
print('Accuracy: {} %'.format(score*100.0))

Accuracy: 95.83333333333334 %
```

</br>
IT WAS AN INTERESTING PROJECT TO WORK ON AND IT HELPED US GAIN A LOT OF EXPERIENCE IN VERY SHORT AMOUNT OF TIME 
SPECIAL CREDITS TO OUR SENIOURS WHO MENTORED AND HELPED US AS WE PROGRESSED.