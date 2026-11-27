#1 import all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#2 load dataset
df= pd.read_csv("salary_data.csv")
#3 understand dataset
df.describe()
df.info()
df.shape
df.head()
df.tail()
df.size
#5 handle missing values
num_cols = df.select_dtypes(include="number").columns
cat_col = df.select_dtypes(include="object").columns
for i in num_cols:
    
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    count = ((df[i] < lower) | (df[i] > upper)).sum()
 
    if count>0:
        df[i]=df[i].fillna(df[i].median())
    elif count==0:
        df[i]=df[i].fillna(df[i].mean())
    else:
        df[i]=df[i].fillna(df[i].mode()[0])
                           
#6 handle the outliers
num_cols = df.select_dtypes(include="number").columns

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = ((df[col] < lower) | (df[col] > upper)).sum()

    print(col, "Outliers:", count)

from sklearn.model_selection import train_test_split
X=df[["Experience"]]
y=df[["Salary"]]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#9 train the model or training data
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
model=reg.fit(X_train,y_train)
model

import joblib
joblib.dump(model,"salary_training_model.pkl")
print("Model saved successfully")

