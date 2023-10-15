import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

app = Flask(__name__)

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()['data']
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    # Extracting the data from the form
    data = [float(request.form[x]) for x in request.form.keys()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    output = regmodel.predict(final_input)[0]
    return render_template('home.html', prediction_text=f'The House Price Prediction is {output}')

if __name__ == '__main__':
    app.run(debug=True)
