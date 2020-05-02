# PROJECT: IPL Result Prediction
Wactching IPL have always been fun and full of enthusiasm. It is probably the most watched domestic cricket tournament in the world. The glamour and the limelight have always attracted millions towards it.
To add spice to this glamourous event , we are presenting a model which will do predictions related to the matches played and give us an idea of the probable results.
We have covered the basic analysis of IPL 2008 to 2017, to get a vivid idea of the vast dataset. The criteria of the data set were:
<br>1)id
<br>2)season
<br>3)city
<br>4)date
<br>5)city
<br>6)toss winner
<br>7)toss decision
<br>8)result
<br>9)dl applied
<br>10)winner
<br>11)win by runs
<br>12)win by wickets
<br>13)player of match
<br>14)venue
<br>15)umpire1
<br>16)umpire2
<br>17)umpire3

# Getting data from kaggle
Kaggle is an online community of data scientists and machine learning practioners offering machine learning competitions, public data paltform and cloud based workbench for data science and artificial intelligence education.

# Data Visualisation 
It is the representation of data in graphical form so that it is more interactable 
The libraries used are :
<br>1)Seaborn 
<br>2)Matplotlib
<br>They can be imported as:
<br>*import seaborn*
<br>*import matplotlib.pyplot*

# Data Preprocessing
It is the technique to transfer raw data into a useful and efficient format.
<br>The libraries used for this were:
<br>1)Numpy
<br>2)Pandas

The data was first cleaned by deleting rows with null values and editing editing duplicate names. Then the text was changed to numeric values because prediction can only be done by the model if they have numeric values.

# Data Splitting
The data was divided into two subsets 
<br>training set- a subset to train a model 
<br>test test- a subset to test the trained model
Make sure the data is large enough to yield statistically meaningful results and the test data is representative of the data set as a whole, in other words, don't pick a test set with different characteristics than the training set. Never train on test data. We used SKLearn 
You can get the library by importing this as-
<br>*from sklearn model_selection import train_test_split*

# Applying Models
For this purpose we used SKLearn library and imported models of our necessity. We used-
<br>1)Logistic Regression
<br>2)Support Vector Machines

In Logistic Regression, outcome of a dependent variable of observation made before hand.
SVM analyses data for classification and regression analysis. It is a supervised learning method that clssifies data into onen of two categories.

# Conclusion
The accuracy we achieved through logistic regression was approximately 61%. On the other hand, we achieved an accuracy of 66% through SVM.
Hence in this case, we can say that SVM is better tahn logistic regression.

# Credits
The project is done by Team "import kaggle as k" of the following members:
<br>1)Ashish Chouhan
<br>2)Shubham Agarwal
<br>3)Shaswat Pandey
<br>4)B.Siddhant