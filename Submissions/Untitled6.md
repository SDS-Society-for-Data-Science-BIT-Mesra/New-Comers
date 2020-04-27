
# PROJECT:- PREDICTION OF AIR POLLUTION IN SEOUL

Prediction of O3 (OZONE) Emission level using the SO2 (SULPHUR DIOXIDE) and NO2 (NITROGEN DIOXIDE) emission data over two years (2017-2019): We used machine learning to predict level of O3(OZONE) in seoul's air. Data of SO2 and NO2 emissions was used to make the predictions on O3. It will tell vulnerability and danger of O3 poisoning in seoul's natural atmosphere. Geological and weather departments conduct such analysis to grab the insights and properly plan their protection mechanism.

# SO BASICALLY WE FOLLOWED BELOW MENTIONED PATTERN IN THIS PARTICULAR PROJECT:

1).COLLECTING AND UPLOADING THE DATASET

2).DATA CLEANING

3).DATA VISUALIZATION

4).DATA PREPROCESSING

5).DATA SPLITTING & MODEL SELECTION

6).CONCLUSION

We started by importing standard packages like numpy,pandas,matplotlib(to make standard plots) and seaborn.

# FIRST STEP: COLLECTING AND UPLOADING DATA ON NOTEBOOK

We downloaded our dataset from Kaggle(world renowned data science platform for people having interest in the field) in form of a *csv file*.

**Link:** https://www.kaggle.com/bappekim/air-pollution-in-seoul

Then we uploaded it to our jupyter notebook by specifying the path and implementing simple panda's syntax :


```python
pd.read_csv('file_name.csv')
```

# SECOND STEP:DATA CLEANING

The very next step by our team was to clean the data or to be precise deal with missing values if any. We simply applied basic codes as **"df.info","df.head()"** and then standard syntax for finding sum of missing values i.e.**"df_summary.isnull().sum())"**. To favour our convenience there were no missing values in the data and hence made our work easy by not worrying about how to deal with them.

# THIRD STEP:DATA VISUALIZATION

To visualize any dataset we mean to plot some *standard distribution graphs* for a generalised view of data because it is well known that pictures speak more than words.
For this we needed tools like **matplotlib** and **seaborn** which are plotting libraries in python.

To install these you we have to run the following code as shown below:


```python
!pip install matplotlib 
```


```python
 !pip install seaborn
```

And import these libraries in our project by using:


```python
import matplotlib.pyplot as plt
```


```python
import seaborn as sns
```

WE CONSIDERED PLOTTING :

*1). A Subplot showing level of all pollutants w.r.t time*

*2). Seaborn's heatmap plot*

*3). Bar Graph plots for each of the pollutants spanning over two years(2017-2019)*

# FOURTH STEP:DATA PREPROCESSING

We had 4 datasets namely: 

*1.Measurement info (showing info about the avg. pollution level of all the pollutants at different stations)*

*2.Measurement station info (showing info about the measurement stations at different places in Seoul)*

*3.Measurement item info (showing info about the indicators of pollution level due to different substances which are   Good,Normal,Bad,Very Bad)*

*4.Measurement summary (it is a summary dataset of all the above 4 datasets)*

To ease the work for us we created a dataset namely **"df_seoul"** consisting of the following labels/columns: 

**a)'SO2'** 

**b)'NO2'**

**c)'O3'**

**d)'CO'**

**e)'PM10'**

**f)'PM2.5'**

using the statement: 




```python
df_seoul = df_summary.groupby(['date'], as_index=False).agg({'SO2':'mean', 'NO2':'mean', 'O3':'mean', 'CO':'mean', 'PM10':'mean', 'PM2.5':'mean'})
```

This dataset contains the average/mean value for each of the pollutants on a particular day in Seoul. *Then for the sake of visualizing the pollutants in bar graphs we divided each of the pollutants in 3 classes based on the the pollution level.*

# FIFTH STEP:MODEL SELECTION AND APPLICATION

We started by selecting our machine learning algorithm to apply which was supposed to be linear regression since the data had much larger dimensions and consequently big enough.

To apply linear regression on our model we need a library that features machine learning algorithms so we imported linear regression from sklearn(scikit learn) which is a standard python library by running a simple code "from sklearn.linear_model import LinearRegression". Next step is to fit the data into this model.

To install sklearn on your computer open your command line and type: 


```python
!pip install sklearn
```

We now move to *split the data into training data and test data.*

To do so run the code below:


```python
from sklearn.model_selection import train_test_split x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
```

We split our data by alloting **30% to the test data and 70% to training data** on which regression is to be applied.

Now we **predict the coefficients of O3(ozone) by using _Multiple Linear Regression_.** The following code shows how we imported and trained our model:


```python
from sklearn.linear_model import LinearRegression 
lr= LinearRegression() 
lr.fit(x_train, y_train)
```

Next, we obtained the *coefficients* of our attributes (SO2, NO2) in a seperate dataframe **'v'**  by:


```python
v=pd.DataFrame(lr.coef_ , index=['Coefficients']).transpose()
```

Finally we predicted the values and put them in a seperate dataframe **'y_pred'** by the following code:


```python
y_pred = lr.predict(x_test) y_pred = pd.DataFrame(y_pred, columns=['Predicted'])
```

# SIXTH STEP: MODEL PERFORMANCE CHECK

Now, our task is to see how well our model performed We imported metrics from sklearn to find **"mean absolute error"**, **"mean squared error"** and **"root mean squared error"** which were coming out **_0.009711303632781035_**, **_0.00021858610968756242_**, **_0.014784657915811322_** respectively.

Have a look at the follwing lines of code:


```python
from sklearn import metrics 
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred)) 
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred)) 
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```

We also found out the **r-square(r2) score** of the model was coming out **0.5755759461710528 or 57.56% (approx.)**. R-Squared is a goodness-of-fit measure for linear regression models. This statistics indicates the percentage of the variance in the dependent variable that the independent variables explain collectively. R-squared measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale. *Usually, the larger the R2, the better the regression model fits your observations.*


```python
metrics.r2_score(y_test,y_pred)
```

A value of 0.5756 suggests that 57.56% of the dependent variable is predicted by the independent variable, and so forth.

# CREDITS:

*THE FOLLOWINNG PROJECT HAS BEEN DONE BY:*

**SUBHANJAN BASU**

**ANJALI VERMA**

**AYUSH MISHRA**

**AUSAF AHMED**

A special thanks to all the mentors of Society For Data Science(SDS) for helping us throughout and mentoring us in this project.


```python

```
