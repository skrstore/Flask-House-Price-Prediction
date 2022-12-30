
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
model = pickle.load(open('HousePricePrediction.model', 'rb'))


@app.route('/')
def index():
    """
    Index route
    """
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict route
    """
    input_data = request.get_json()
    location = input_data.get('location')
    bhk = float(input_data.get('bhk'))
    bath = float(input_data.get('bath'))
    total_sqft = input_data.get('total_sqft')

    input_df = pd.DataFrame([[location, total_sqft, bath, bhk]], columns=[
                            'location', 'total_sqft', 'bath', 'bhk'])
    prediction = model.predict(input_df)[0] * 1e5

    return {'message': "success", "prediction_amount": np.round(prediction, 2)}


if __name__ == '__main__':
    app.run(debug=True, port=8080)
