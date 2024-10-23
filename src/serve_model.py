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

# Load the trained model
model = joblib.load('random_forest_model.pkl')

@app.route('/')
def index():
    app.logger.info("Home page accessed.")
    return render_template('index.html')

@app.route('/test_logging')
def test_logging():
    app.logger.info("Test logging route accessed.")
    return "Check the log for this message!"

def make_prediction(data):
    features = [
        data["Time"],
        data["V1"],
        data["V2"],
        data["V3"],
        data["V4"],
        data["V5"],
        data["V6"],
        data["V7"],
        data["V8"],
        data["V9"],
        data["V10"],
        data["V11"],
        data["V12"],
        data["V13"],
        data["V14"],
        data["V15"],
        data["V16"],
        data["V17"],
        data["V18"],
        data["V19"],
        data["V20"],
        data["V21"],
        data["V22"],
        data["V23"],
        data["V24"],
        data["V25"],
        data["V26"],
        data["V27"],
        data["V28"],
        data["Amount"]
    ]

    features_df = pd.DataFrame([features], columns=[
         "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
        "V10", "V11", "V12", "V13", "V14", "V15", "V16",
        "V17", "V18", "V19", "V20", "V21", "V22", "V23",
        "V24", "V25", "V26", "V27", "V28", "Amount"
    ])

    app.logger.info("Features for prediction: %s", features_df.to_dict(orient='records'))
    prediction = model.predict(features_df)
    return int(prediction[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    app.logger.info("Received data: %s", data)
    
    try:
        prediction = make_prediction(data)
        app.logger.info("Prediction made: %d", prediction)
        return jsonify({'prediction': prediction})
    except Exception as e:
        app.logger.error("Error during prediction: %s", str(e))
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    app.logger.info("Starting Flask application.")
    app.run(host='0.0.0.0', port=5001)
