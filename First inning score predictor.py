#!/usr/bin/env python
# coding: utf-8

# In[22]:




import pandas as pd
import numpy as np
import pickle#Pickle in Python is primarily used in serializing and deserializing a Python object.
import matplotlib.pyplot as plt





data = pd.read_csv("ipl.csv")#Loading the data



data.columns = ['mid', 'date', 'venue', 'bat_team', 'bowl_team', 'batsman', 'bowler',
       'runs', 'wickets', 'overs', 'runs_last_6', 'wickets_last_6', 'striker',
       'non-striker', 'total']
data.head()





data = data.drop(['batsman','bowler','mid','striker','non-striker'],axis=1).copy()





data.shape





data['bat_team'].unique()





print("Data_shape:",data.shape)


# # considering the only teams which are Playing the IPL in Current seasons

consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']#removing inconsistent teams and selecting only consistent_teams from  the dataset





# Keeping only consistent teams
print('Before removing inconsistent teams: {}'.format(data.shape))
data = data[(data['bat_team'].isin(consistent_teams)) & (data['bowl_team'].isin(consistent_teams))]
print('After removing inconsistent teams: {}'.format(data.shape))


# Removing the first 6 overs data in every match
print('Before removing first 6 overs data: {}'.format(data.shape))
data = data[data['overs']>=6.0]
print('After removing first 6 overs data: {}'.format(data.shape))




# Converting the date Column form string into Data time object 




print("Before converting date column from string to datetime object: {}".format(type(data.iloc[0,0])))






# Converting the column 'date' from string into datetime object
from datetime import datetime
print("Before converting 'date' column from string to datetime object: {}".format(type(data.iloc[0,0])))
data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
print("After converting 'date' column from string to datetime object: {}".format(type(data.iloc[0,0])))





data.corr()





corr_matrix = data.corr()
top_corr_features = corr_matrix.index





top_corr_features





import seaborn as sns
plt.figure(figsize=(13,10))

g = sns.heatmap(data=data[top_corr_features].corr(), annot=True, cmap='RdYlGn')


# # Data Preprocessing



encoded_data = pd.get_dummies(data = data, columns= ["bat_team",'bowl_team'])





encoded_data .columns




encoded_data.head()





ncoded_data= encoded_data[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_6', 'wickets_last_6', 'total']]





ncoded_data.shape





ncoded_data.head()


# # Splitting the dataset into Training data and Test data

# # dt can be used to access the values of the series as datetimelike and return several properties. Pandas Series. dt. minute attribute return a numpy array containing the minutes of the datetime in the underlying data of the given series object.




X_train = ncoded_data.drop("total",axis=1)[ncoded_data['date'].dt.year <=2016]
X_test = ncoded_data.drop("total",axis =1)[ncoded_data['date'].dt.year >= 2017]
X_val = ncoded_data.drop("total",axis=1)[ncoded_data['date'].dt.year >=2017]




Y_train = ncoded_data[encoded_data['date'].dt.year <= 2016]['total'].values
Y_test = ncoded_data[encoded_data['date'].dt.year >= 2017]['total'].values
Y_val = ncoded_data[encoded_data['date'].dt.year >= 2017]['total'].values





X_train = X_train.drop("date",axis=1)
X_test = X_test.drop("date",axis = 1)
X_val = X_val.drop("date",axis=1)



#from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()   #normalising the data
#X_train = min_max_scaler.fit_transform(X_train)



X_train



print("X_train shape:",X_train.shape)
print("X_test shape:",X_test.shape)
print("Y_train_shape",Y_train.shape)
print("Y_test_shape:",Y_test.shape)
print("X_val_shape:",X_val.shape)
print("Y_val_shape:",Y_val.shape)


# # X_train (21 input features, 70% of full dataset)
#  # Y_train (1 label, 70% of full dataset)
#  # X_test (21 input features, 15% of full dataset)
#  # Y_test (1 label, 15% of full dataset)
# # X_val (21 input features, 15% of full dataset)
# # Y_val (1 label, 15% of full dataset)
# 




from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train,Y_train)







# In[23]:


#creating a pickle file for  the calssifier
filename = 'Ipl_first_innings_score_prediction_model.pkl'
pickle.dump(linear_regressor,open(filename,'wb'))



# In[ ]:




