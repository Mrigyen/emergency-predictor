#!/usr/bin/env python

from flask import Flask
import model
import encoding
import proximity

app = Flask(__name__)


@app.route("/")
def status() -> str:
    status = str("Model trained: " + str(model.check_for_saved_model()))
    return status


@app.route("/predict/<str:military_time>/<str:lat>/<str:longitude>/<int:age>/<int:gender>")
def predict(military_time, lat, longitude, age, gender) -> float:
    # get proximity score
    location_node = [float(lat), float(longitude)]
    proximity_score = proximity.get_proximity(location_node)

    # get prediction of emergency
    sin_time = encoding.sin_time(encoding.military_time_in_minutes_fn(military_time))
    cos_time = encoding.cos_time(encoding.military_time_in_minutes_fn(military_time))
    prediction = model.predict(age, gender, sin_time, cos_time)
   
    # return multiplication
    return proximity_score * prediction


# run it directly via python3 main.py
if __name__ == "__main__":
    app.run()
