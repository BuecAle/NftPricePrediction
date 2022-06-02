# Taken from https://www.kaggle.com/code/subhradeep88/house-price-predict-decision-tree-random-forest/notebook

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import confusion_matrix

import parameter
import warnings
warnings.filterwarnings('ignore')

# import dataset from csv-File
# file path is stored in parameter.py
dataset = pd.read_csv(parameter.data_analysation.file)

df = dataset[["asset_id", "price_usd", "name", "owner", "seller", "buyer", "burnable", "date", "media", "coll_name",
        "coll_author", "coll_market_fee"]]

print(df.head())

# transform categorical variables into one-hot-encoding
ord_enc = OrdinalEncoder()
df["name"] = ord_enc.fit_transform(dataset[["name"]])
df["owner"] = ord_enc.fit_transform(dataset[["owner"]])
df["seller"] = ord_enc.fit_transform(dataset[["seller"]])
df["buyer"] = ord_enc.fit_transform(dataset[["buyer"]])
df["burnable"] = df["burnable"].astype(int)
df["date"] = pd.to_numeric(dataset["date"].str.replace('-', ''))
df["coll_name"] = ord_enc.fit_transform(dataset[["coll_name"]])
df["coll_author"] = ord_enc.fit_transform(dataset[["coll_author"]])

print(df.info())

#Lets find out how many unique values are present in each column
for value in df:
    print('For {},{} unique values present'.format(value,df[value].nunique()))

df = df.drop(['asset_id','media'],axis=1)

print(df.head())
print(df.corr())

# plt.figure(figsize=(10,6))
# sns.plotting_context('notebook',font_scale=1.2)
# g = sns.pairplot(df[["name", "owner", "seller", "buyer", "date", "price_usd", "coll_name",
#          "coll_author", "coll_market_fee"]], hue='price_usd',size=4)
# g.set(xticklabels=[])
# plt.show()

sns.jointplot(x='name',y='price_usd',data=df,kind='reg',size=4)
sns.jointplot(x='owner',y='price_usd',data=df,kind='reg',size=4)
sns.jointplot(x='seller',y='price_usd',data=df,kind='reg',size=4)
sns.jointplot(x='buyer',y='price_usd',data=df,kind='reg',size=4)
sns.jointplot(x='date',y='price_usd',data=df,kind='reg',size=4)
sns.jointplot(x='coll_author',y='price_usd',data=df,size=4)
sns.jointplot(x='coll_name',y='price_usd',data=df,kind='reg',size=4)
sns.jointplot(x='coll_market_fee',y='price_usd',data=df,size=4)

plt.figure(figsize=(15,10))
columns =["name", "owner", "seller", "buyer", "date", "price_usd", "coll_name", "coll_author", "coll_market_fee"]
sns.heatmap(df[columns].corr(),annot=True)

# X(Independent variables) and y(target variables)
X = df.iloc[:,1:].values
y = df.iloc[:,0].values

#Splitting the data into train,test data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

mlr = LinearRegression()
mlr.fit(X_train,y_train)
mlr_score = mlr.score(X_test,y_test)
pred_mlr = mlr.predict(X_test)
expl_mlr = explained_variance_score(pred_mlr,y_test)

tr_regressor = DecisionTreeRegressor(random_state=0)
tr_regressor.fit(X_train,y_train)
tr_regressor.score(X_test,y_test)
pred_tr = tr_regressor.predict(X_test)
decision_score=tr_regressor.score(X_test,y_test)
expl_tr = explained_variance_score(pred_tr,y_test)

rf_regressor = RandomForestRegressor(n_estimators=28,random_state=0)
rf_regressor.fit(X_train,y_train)
rf_regressor.score(X_test,y_test)
rf_pred =rf_regressor.predict(X_test)
rf_score=rf_regressor.score(X_test,y_test)
expl_rf = explained_variance_score(rf_pred,y_test)


print("Multiple Linear Regression Model Score is ",round(mlr.score(X_test,y_test)*100))
print("Decision tree  Regression Model Score is ",round(tr_regressor.score(X_test,y_test)*100))
print("Random Forest Regression Model Score is ",round(rf_regressor.score(X_test,y_test)*100))

#Let's have a tabular pandas data frame, for a clear comparison

models_score =pd.DataFrame({'Model':['Multiple Linear Regression','Decision Tree','Random forest Regression'],
                            'Score':[mlr_score,decision_score,rf_score],
                            'Explained Variance Score':[expl_mlr,expl_tr,expl_rf]
                           })
models_score.sort_values(by='Score',ascending=False)




















































