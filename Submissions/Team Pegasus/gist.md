# PROJECT:- 
In our project we tried to analyze and visualize the basic features that affects the life expectancy of people living in India.
It is a basic Data Analysis project.

### Team members:-
1. SOUVIK DEY
2. RAAVEE MAHESWARI
3. TUSHAR RAJ and
4. HARSHIT MISHRA

*SUMMARY OF OUR PROJECT ARE AS FOLLOWS:-*
1. Importing the dataset from kaggle(Link :https://www.kaggle.com/kumarajarshi/life-expectancy-who)
2. Extracting the data of India from our dataset, using

                df.loc[] 
    function.

3. Using simple visualization technique from libriaries like seaborn and matplotlib, to understand the trends of the features.

## IMPORTING USEFUL LIBRIARIES:-
1. numpy
   
            import numpy as np

2. pandas
   
            import pandas as pd

3. matplotlib and seaborn
      
            import matplotlib.pyplot as pyplot
            import seaborn as sns

## IMPORTING OUR DATASET FROM KAGGLE:-
  
  Link for our dataset: https://www.kaggle.com/kumarajarshi/life-expectancy-who
  
  Using our pandas libriary for accessing our dataset
             
             pd=df.read_csv() function.

## DATA-CLEANING:-
 We checked for the null values in our extracted dataset(India_data) using
               
               India_data.isnull() 
 
function which replaced all the null values with true and all non-nul values with false.We also used 
               
               India_data.isnull().sum() 
    
function to calculate total number of null values in our dataset.After that we replaced the null values of that feature(HepB) with mean of that column using 
               
               India_data.HepB=India_data.HepB.fillna(India_data.HepB.mean()) function.


## Data-visualization:-

 We used some basic visualization techniques like barplot and line plot imported from our matplotlib and seaborn libriaries,
 to understand the trends of the feature which were affecting the life expectancy of India.
 Some basic codes from our visualization part are as follows:
 
 ### FOR BARPLOT

            plt.figure(figsize=(14,6),dpi=100)

            sns.barplot(A1['Year'],A1['Life expectancy '],palette='icefire')
            plt.title('YEAR WISE LIFE EXPECTANCY IN INDIA',size=20)
            plt.legend()
            plt.grid(color='grey',axis='y')
            plt.tight_layout()
   

 ### FOR LINE GRAPH
 
           plt.figure(figsize=(16,6),dpi=100)

           plt.plot(A['Year'],A['Polio'],color='orange',label='POLIO IMMUNIZATION',marker='o')
           plt.plot(A['Year'],A['HepB'],color='blue',label='HEPATITS B IMMUNIZATION',marker='o')

           plt.grid(color='grey')
           plt.legend()
           plt.tight_layout()

And many more are there.

WE ALSO CREATED A CORRELATION MATRIX TO SEE HOW OTHER FEATURES LIKE POLIO IMMUNIZATION, ADULT MORTALITY RATE, 
INFANT DEATHS ARE RELATED WITH THE LIFE EXPECTANCY OF PEOPLE LIVING IN INDIA.
WE ALSO PLOTTED A HEATMAP TO UNDERSTAND OUR MATRIX BETTER THROUGH VISUALIZING IT, BELOW IS THE CODE FOR SAME:

       plt.figure(figsize=(40,20),dpi=100)
       sns.heatmap(corr,annot=True,square=True,linewidths=.5)
       plt.title('CORRELATION HEATMAP',size=20)


*IT WAS AN INTERSTING PROJECT TO WORK ON, WE WOULD LIKE TO THANK OUR SENIORS AND MENTORS WHO HELPED US IN EVERY STEP.* 