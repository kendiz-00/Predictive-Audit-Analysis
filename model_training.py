import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model():
    # Load the data
    df = pd.read_csv('audit_data_short.csv')
    
    # Select features
    features = ['Sector_score', 'PARA_A', 'Score_A', 'PARA_B', 'Score_B', 
               'TOTAL', 'numbers', 'Money_Value', 'District_Loss']
    
    X = df[features]
    y = df['Risk']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    joblib.dump(rf_model, 'models/model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Calculate and print metrics
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return rf_model, scaler

if __name__ == "__main__":
    train_model() 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model():
    # Load the data
    df = pd.read_csv('audit_data_short.csv')
    
    # Select features
    features = ['Sector_score', 'PARA_A', 'Score_A', 'PARA_B', 'Score_B', 
               'TOTAL', 'numbers', 'Money_Value', 'District_Loss']
    
    X = df[features]
    y = df['Risk']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    joblib.dump(rf_model, 'models/model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Calculate and print metrics
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return rf_model, scaler

if __name__ == "__main__":
    train_model() 