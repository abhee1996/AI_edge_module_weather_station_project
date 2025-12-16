import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import joblib

# Load preprocessed data
X_train = pd.read_csv('X_train.csv').values
y_train = pd.read_csv('y_train.csv').values.ravel()
X_test = pd.read_csv('X_test.csv').values
y_test = pd.read_csv('y_test.csv').values.ravel()


# # Model Neural Network
model = Sequential()
model.add(Dense(64, input_dim=3, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1)) 
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
nn_pred = model.predict(X_test)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))
nn_mae = mean_absolute_error(y_test, nn_pred)
print(f'Neural Network - RMSE: {nn_rmse}, MAE: {nn_mae}')

models = {'nn': (model, nn_rmse)}
best_key, (best_NN_model, _) = min(models.items(), key=lambda x: x[1][1])
if best_key == 'nn':
    best_NN_model.save('NN_model.h5')
print(f'Best model saved as {"NN_model.h5"}')