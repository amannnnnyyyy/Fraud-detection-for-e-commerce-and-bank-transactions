from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('models/random_forest_model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    
    # Make prediction
    prediction = model.predict(df)
    
    # Convert the prediction to standard Python int
    prediction_value = int(prediction[0])  # Convert to int to ensure it's JSON serializable
    
    return jsonify({'prediction': prediction_value})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
