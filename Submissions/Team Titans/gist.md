#                   PROJECT:: WALMART SALES PREDICTION
Sales forecasting is estimating what a company’s future sales are likely to be , based on the sales records as well as the market research . Information used for sales forecasting must be well organized and may include information on the competition and the statistics that affect the businesses customer base . Companies conduct sales forecasting in hopes of identifying the patterns so that the revenue and the  cash flow can be maximized.

[SALES FORCATING](https://www.ukessays.com/essays/marketing/what-is-sales-forecasting-marketing-essay.php)
 ___
 ## We did our project according to following pattern::
 1. Getting data from kaggle
 1. Visualising the data
 1. Data preprocessing
 1. Data splitting
 1. Applying Models
 1. Conclusion about which model performed the best
 
>## Getting data from kaggle  

Kaggle is the world's largest data science community with powerful tools and resources to help you achieve your data science goals.
 
 [WALMART SALES](https://www.kaggle.com/anshg98/walmart-sales)
 
>## Visualising the data  
 Data visualization is the graphical representation of information and data.    
 The Library we used  
    (i) Seaborn  
    (ii) Matplotlib   
   you can get these libraries by running following on your Jupyter Notebook.
   
   ```
      !pip install seaborn
      !pip install matplotlib
   ```
  and import them as given below:  
  ```python
    import seaborn as sns
    import matplotlib.pyplot as plt
  ```
  >## Data preprocessing
  Data preprocessing is a data mining technique which is used to transform the raw data in a useful and efficient format.  
  [Data Preprocessing](https://towardsdatascience.com/data-preprocessing-concepts-fa946d11c825)  
  Now was the time to check null values
We deleted those columns as the missing values were quite high. 
The data not being categorical was used through different models. The accuracy obtained was highly less than required. Therefore we then used PANDAS library to convert columns into categorical data: 
We also used FOR LOOP to convert our prediction into categorical as every input we used was categorical.
This will help model to predict more accurately.
  The Library we used   
     (i) Numpy  
    (ii) Pandas   
   you can get these libraries by running following on your Jupyter Notebook.
   
   ```
      !pip install numpy
      !pip install pandas
   ```
  and can be imported as given below:  
  ```python
    import numpy as np
    import pandas as pd
  ```
   >## Data splitting  
   As we work with datasets, a machine learning algorithm works in two stages. We usually split the data around 20%-80% between testing and training stages. Under supervised learning, we split a dataset into a training data and test data in Python ML.  
   [Data splitting](https://data-flair.training/blogs/train-test-set-in-python-ml/)
    The Library we used in this Project is Sklearn
    
     you can get these libraries by running following on your Jupyter Notebook.
   
   ```
      !pip install sklearn
   ```
  and run the following code as given below:  
  ```python  
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
  ```
  As of now we understood that the task is to predict categorical data.
Hence that’s a classification	 problem.
 >## Applying Models
 For this purpose we used SKLEARN LIBRARY and	 imported our desired models.  
We  used following models:  
1. Logistic Regressiom
1. Naive Bayes
1. K Nearest Neighbour
1. Decision Tree

the following code is an example of how we imported to how we trained our model::  
  ```python  
     from sklearn.linear_model import LogisticRegression
  ```
   
  ```python  
     LR=LogisticRegression()
  ```
  ```python  
     LR.fit(X_train,y_train)
  ```
 >## Conclusion about which model performed the best
 To compare which model perform well on our dataset we used accuracy_score. The model with the maximum accuracy was suitably chosen.   
Hence we drawn the conclusion.  

LOOK AT THESE CODES:

  ```python  
     y_predict=LR.predict(X_test)
  ```
   
  ```python  
     from sklearn.metrics import accuracy_score
  ```
  ```python  
     accuracy_score(y_test,y_predict)
  ```
  The above lines of code are an example of how we calculated Accuracy Score.
  We have to calculate this for every model and finally we compare them to draw conclusion.
  
  ___
  ## Credits:
  The project is done by Team of following members::  
  **[Ankur Singh](https://github.com/update-ankur)**           
  **[Siddharth Khan](https://github.com/sidd-tech)**  
  **[Yash khandelwal](https://github.com/YashK07)**  
  **[Kaivalyamani Tripathi](https://github.com/Kaivalya4)**    
  
