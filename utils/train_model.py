"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import Lasso

# Fetch training data and preprocess for modeling
train_raw = pd.read_csv('Train.csv') 
riders_raw = pd.read_csv('Riders.csv')
train = train_raw.merge(riders_raw, how="left", on = "Rider Id")

ys_train = train[['Time from Pickup to Arrival']]
Xs_train = train[['Pickup Lat','Pickup Long',
                 'Destination Lat','Destination Long']]

# Fit model
lasso = Lasso(alpha=0.01)
print ("Training Model...")
lasso.fit(Xs_train, ys_train)

# Pickle model for use within our API
save_path = '../trained-models/RForrest_model.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lasso, open(save_path,'wb'))
