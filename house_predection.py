import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('kc_house_data.csv')

data.drop('id', inplace=True, axis=1)
data['date'] = pd.to_datetime(data['date'])
data['Month'] = data['date'].apply(lambda date: date.month)
data['Year'] = data['date'].apply(lambda date: date.year)
data['bathrooms'] = np.round(data['bathrooms'])
data['floors'] = np.round(data['floors'])

data.isnull().sum()
data.dropna(inplace=True)

data.drop('date', inplace=True, axis=1)
data.isnull().sum()

X = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
          'grade', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']].values
y = data['price'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=50)
# std = StandardScaler()
# X = std.fit_transform(X)
rfr = RandomForestRegressor(n_estimators=200)
rfr.fit(X_train, y_train)
score_rfr = rfr.score(X_train, y_train)
score_rfr


x_new = np.array([3, 1.0, 1180, 5650, 1.0, 0, 0, 3, 7,
                 1180, 0, 1340, 5650]).reshape(-1, 1)
# std = StandardScaler()
# x_new = std.fit_transform(x_new)
rfr.predict(x_new)


# with open("house_prediction.model", "wb") as f:
#     pickle.dump(model, f)


# with open("house_prediction.model", "rb") as f:
#     mp = pickle.load(f)
