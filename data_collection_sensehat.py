from sense_hat import SenseHat
import csv
import time
from datetime import datetime

sense = SenseHat()

csv_file = 'weather_data.csv'
header = ['timestamp', 'temperature', 'humidity', 'pressure']

with open(csv_file, 'a', newline='') as f:
    writer = csv.writer(f)
    if f.tell() == 0: 
        writer.writerow(header)

while True:
    timestamp = datetime.now().isoformat()
    temperature = sense.get_temperature()
    humidity = sense.get_humidity()
    pressure = sense.get_pressure()  

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, temperature, humidity, pressure])

    time.sleep(10)  