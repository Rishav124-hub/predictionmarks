                                    # Importing neccessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
                                    # Reading the dataset from given URL link
table=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
                                    # ploting graph of Hours vs Scores
table.plot(x='Hours',y='Scores', style='o')
plt.title('Hours vs Scores')
plt.xlabel('Hours Studies ')
plt.ylabel('Scores')
plt.show()
                                    # Spliting data in independent variable
X= table["Hours"].values.reshape(-1,1)
Y= table["Scores"].values.reshape(-1,1)
                                    #Now we do train/test split
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train, Y_test =train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
                                    #plot the predicated fit line
b=regressor.intercept_
m=regressor.coef_
plt.scatter(X,Y)
plt.plot(X,m*X+b)
plt.title('Hours vs Score')
plt.xlabel('no.of hoursstudied')
plt.ylabel('Scores')
plt.show()
                                    #Function for you predicated score
def score_predictor(Hours):
    b=regressor.intercept_
    m=regressor.coef_
    if (0<=Hours<=24):
        Hours=float(Hours)
        Hours=np.array(Hours).reshape(-1,1)
        Score=np.round(m*Hours+b)
        Score =100 if Score >100 else Score
        print(f"You have studied for {float(Hours)} hrs and It's predicated you will get {int(Score)} score")
    else:
        print("Enter a valid number")

score_predictor(float(input("Enter how many hours you have studied in a day = ")))