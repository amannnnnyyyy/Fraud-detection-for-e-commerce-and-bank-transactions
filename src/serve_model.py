from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import logging

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Set up basic logging configuration
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app.logger.info("Logging is set up correctly.")

# Load both models
ecommerce_model = joblib.load('random_forest_model+fraud.pkl')
bank_model = joblib.load('random_forest_model+credit.pkl')

def make_prediction(model, data):
    features = [data[key] for key in sorted(data.keys())]  # Assuming data keys match the model input order
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)
    return int(prediction[0])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ecommerce_predict', methods=['POST'])
def ecommerce_predict():
    data = request.get_json()
    try:
        prediction = make_prediction(ecommerce_model, data)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': 'E-commerce prediction error.'}), 500

@app.route('/bank_predict', methods=['POST'])
def bank_predict():
    data = request.get_json()
    try:
        prediction = make_prediction(bank_model, data)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': 'Bank prediction error.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
