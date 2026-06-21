from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os

app = Flask(__name__, template_folder=os.path.dirname(os.path.abspath(__file__)))

# Load trained model
try:
    model = pickle.load(open("model.pkl", "rb"))
except FileNotFoundError:
    print("Error: model.pkl not found in the fraud directory")
    model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please train and save the model first."})
    
    try:
        data = request.get_json()
        
        # Extract all 30 features (V1 to V28 + Time + Amount)
        features = np.array([
            float(data.get('Time', 0)),
            float(data.get('V1', 0)),
            float(data.get('V2', 0)),
            float(data.get('V3', 0)),
            float(data.get('V4', 0)),
            float(data.get('V5', 0)),
            float(data.get('V6', 0)),
            float(data.get('V7', 0)),
            float(data.get('V8', 0)),
            float(data.get('V9', 0)),
            float(data.get('V10', 0)),
            float(data.get('V11', 0)),
            float(data.get('V12', 0)),
            float(data.get('V13', 0)),
            float(data.get('V14', 0)),
            float(data.get('V15', 0)),
            float(data.get('V16', 0)),
            float(data.get('V17', 0)),
            float(data.get('V18', 0)),
            float(data.get('V19', 0)),
            float(data.get('V20', 0)),
            float(data.get('V21', 0)),
            float(data.get('V22', 0)),
            float(data.get('V23', 0)),
            float(data.get('V24', 0)),
            float(data.get('V25', 0)),
            float(data.get('V26', 0)),
            float(data.get('V27', 0)),
            float(data.get('V28', 0)),
            float(data.get('Amount', 0))
        ]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        return jsonify({
            "prediction": int(prediction),
            "fraud_probability": float(probability[1]),
            "legitimate_probability": float(probability[0])
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)