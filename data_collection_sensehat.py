from sense_hat import SenseHat
import csv
import time
from datetime import datetime

sense = SenseHat()

# Create or append to CSV file
csv_file = 'weather_data.csv'
header = ['timestamp', 'temperature', 'humidity', 'pressure']

with open(csv_file, 'a', newline='') as f:
    writer = csv.writer(f)
    if f.tell() == 0:  # Write header if file is empty
        writer.writerow(header)

# Collect data every 60 seconds (adjust as needed)
while True:
    timestamp = datetime.now().isoformat()
    temperature = sense.get_temperature()
    humidity = sense.get_humidity()
    pressure = sense.get_pressure()  # In hPa (millibars)

    # Append to CSV
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, temperature, humidity, pressure])

    time.sleep(10)  # Collect every minute