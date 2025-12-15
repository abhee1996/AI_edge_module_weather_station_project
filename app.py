

from flask import Flask, render_template, jsonify
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sense_hat import SenseHat
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib

# Load scaler and model 
scaler = joblib.load('scaler.pkl')
model = tf.keras.models.load_model('best_nn_model.h5')  

sense = SenseHat()

# Use the condition_to_weather
def condition_to_weather(pred):
    pred = round(pred)
    if pred == -4: return  "Severe/Moderate"
    elif pred == -3: return "Cloudy"
    elif pred == -2: return "Thunderstorms"
    elif pred == -1: return "hot"
    elif pred == -0: return "warm"

    elif pred == 0: return "clear"
    elif pred == 1: return "Cold"
    elif pred == 2: return "Severe cold"
    elif pred == 3: return "Freezing"
    elif pred == 4: return "Snow"
    elif pred == 5: return  "very Snow"
    else: return "Unknown"

def predict_condition():
    temperature = sense.get_temperature()
    humidity = sense.get_humidity()
    pressure = sense.get_pressure()
    input_data = np.array([[temperature, humidity, pressure]])
    print('input_data',input_data)
    input_scaled = scaler.transform(input_data)
    print('input_scaled',input_scaled)
    prediction = model.predict(input_scaled)[0][0]  # Adjust for model type
    print('pred',prediction)
    weather_str = condition_to_weather(prediction)
    print('weather_str',weather_str)
    message = f"Temp: {temperature:.1f}C Hum: {humidity:.1f}% Press: {pressure:.1f}hPa Pred: {weather_str}"
    sense.show_message(message, scroll_speed=0.1, text_colour=(255, 255, 255))

    return prediction, weather_str, {'temp': temperature, 'hum': humidity, 'press': pressure}

app = Flask(__name__)

@app.route('/')
def index():
    pred_num, pred_str, data = predict_condition()
    return render_template('index.html', data=data, pred_num=pred_num, pred_str=pred_str)

@app.route('/plot')
def plot():
    _, _, data = predict_condition()  # Get fresh data
    fig, ax = plt.subplots()
    ax.bar(['Temp', 'Hum', 'Press'], [data['temp'], data['hum'], data['press']])
    ax.set_ylabel('Values')
    ax.set_title('Current Weather Data')
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return f'<img src="data:image/png;base64,{plot_url}">'

@app.route('/api/data')
def api_data():

    # In app.py, api_data route
    pred_num, pred_str, data = predict_condition()
    pred_num = float(pred_num)  # Convert np.float32 to float
    return jsonify({'data': data, 'pred_num': pred_num, 'pred_str': pred_str})
    # pred_num, pred_str, data = predict_condition()
    # return jsonify({'data': data, 'pred_num': pred_num, 'pred_str': pred_str})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
