from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from model_training import train_model
import os

app = Flask(__name__)

# Load model and scaler if they exist, otherwise train new ones
if not os.path.exists('models/model.pkl'):
    os.makedirs('models', exist_ok=True)
    model, scaler = train_model()
else:
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get values from the form
            features = {
                'Sector_score': float(request.form['sector_score']),
                'PARA_A': float(request.form['para_a']),
                'Score_A': float(request.form['score_a']),
                'PARA_B': float(request.form['para_b']),
                'Score_B': float(request.form['score_b']),
                'TOTAL': float(request.form['total']),
                'numbers': float(request.form['numbers']),
                'Money_Value': float(request.form['money_value']),
                'District_Loss': float(request.form['district_loss'])
            }
            
            # Create DataFrame with features
            df = pd.DataFrame([features])
            
            # Scale features
            scaled_features = scaler.transform(df)
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            probability = model.predict_proba(scaled_features)[0]
            
            result = {
                'prediction': int(prediction),
                'probability': float(probability[1]),
                'risk_level': 'High Risk' if prediction == 1 else 'Low Risk'
            }
            
            return render_template('results.html', result=result)
            
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True) 
import pandas as pd
import numpy as np
import joblib
from model_training import train_model
import os

app = Flask(__name__)

# Load model and scaler if they exist, otherwise train new ones
if not os.path.exists('models/model.pkl'):
    os.makedirs('models', exist_ok=True)
    model, scaler = train_model()
else:
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get values from the form
            features = {
                'Sector_score': float(request.form['sector_score']),
                'PARA_A': float(request.form['para_a']),
                'Score_A': float(request.form['score_a']),
                'PARA_B': float(request.form['para_b']),
                'Score_B': float(request.form['score_b']),
                'TOTAL': float(request.form['total']),
                'numbers': float(request.form['numbers']),
                'Money_Value': float(request.form['money_value']),
                'District_Loss': float(request.form['district_loss'])
            }
            
            # Create DataFrame with features
            df = pd.DataFrame([features])
            
            # Scale features
            scaled_features = scaler.transform(df)
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            probability = model.predict_proba(scaled_features)[0]
            
            result = {
                'prediction': int(prediction),
                'probability': float(probability[1]),
                'risk_level': 'High Risk' if prediction == 1 else 'Low Risk'
            }
            
            return render_template('results.html', result=result)
            
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True) 