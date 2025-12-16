from sense_hat import SenseHat
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib

sense = SenseHat()
scaler = joblib.load('scaler.pkl')  

model = tf.keras.models.load_model('NN_model.h5')


def condition_to_weather(pred):
    pred_rounded = round(pred)
    if pred_rounded == 0: return "Clear"
    elif pred_rounded == 1: return "Cloudy"
    elif pred_rounded == -1: return "Fog/Haze"
    elif pred_rounded == 3: return "Drizzle"
    elif pred_rounded == 4: return "Rain"
    elif pred_rounded == -5: return "Snow"
    elif pred_rounded == -6: return "Freezing/Ice"
    elif pred_rounded == 7: return "Thunderstorms"
    elif pred_rounded in [8, 9]: return "Severe/Moderate"
    elif pred_rounded in [-8, -9]: return "Extreme Cold/Severe"
    else: return "Unknown"


def predict_condition(model):
    temperature = sense.get_temperature()
    humidity = sense.get_humidity()
    pressure = sense.get_pressure()
    input_data = np.array([[temperature, humidity, pressure]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]
    weather_str = condition_to_weather(prediction)
    return prediction, weather_str, {'temp': temperature, 'hum': humidity, 'press': pressure}


pred_num, pred_str, data = predict_condition(model)
print(f'Predicted numeric: {pred_num}, Weather: {pred_str}, Data: {data}')