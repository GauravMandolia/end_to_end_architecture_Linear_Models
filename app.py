import pickle
import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load Model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')  # First Page of HTML...


@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()  # Use get_json() instead of json()
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])  # Output is 2-dimensional, so take the first element only.


if __name__ == "__main__":
    app.run(debug=True)
