#!/usr/bin/env python

from flask import Flask
import model

app = Flask(__name__)


@app.route("/")
def status() -> str:
    status = str("Model trained: " + str(model.check_for_saved_model()))
    return status


@app.route("/predict/<str:military_time>/<str:lat>/<str:long>/<int:age>/<int:gender>")
def predict() -> float:
    pass


# run it directly via python3 main.py
if __name__ == "__main__":
    app.run()
