from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # Allow all origins

# Load your models
bank_model = joblib.load('random_forest_model+credit.pkl')
ecommerce_model = joblib.load('random_forest_model+fraud.pkl')

# Route to serve the HTML template
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/bank_predict', methods=['POST'])
def bank_predict():
    try:
        # Get JSON data from the request
        data = request.json
        
        # Convert JSON to DataFrame
        df = pd.DataFrame([data])
        
        # Make predictions
        prediction = bank_model.predict(df)
        
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/ecommerce_predict', methods=['POST'])
def ecommerce_predict():
    try:
        # Get JSON data from the request
        data = request.json
        
        # Convert JSON to DataFrame
        df = pd.DataFrame([data])
        
        # Make predictions
        prediction = ecommerce_model.predict(df)
        
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
