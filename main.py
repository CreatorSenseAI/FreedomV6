import os
import logging
import json
import numpy as np
import pandas as pd
import random
import time
import requests
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from cryptography.fernet import Fernet

# Setup Logging
logging.basicConfig(filename='freedom_v6.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting Freedom V6...")

# Load Configurations
CONFIG_FILE = "config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
else:
    config = {}
    logging.warning("Config file missing. Using defaults.")

# Flask API
app = Flask(__name__)

# Encryption Setup
key = Fernet.generate_key()
cipher = Fernet(key)
logging.info(f"Generated encryption key: {key.decode()} - Save this securely!")

# Load Models
revenue_model_path = config.get("revenue_model", "models/lstm_revenue_weights.h5")
fan_model_path = config.get("fan_model", "models/lstm_fan_weights.h5")

if os.path.exists(revenue_model_path):
    revenue_model = load_model(revenue_model_path)
    logging.info("Revenue model loaded.")
else:
    revenue_model = None
    logging.warning("Revenue model not found!")

if os.path.exists(fan_model_path):
    fan_model = load_model(fan_model_path)
    logging.info("Fan model loaded.")
else:
    fan_model = None
    logging.warning("Fan model not found!")

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "Freedom V6 API Running"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        earnings = float(data.get("earnings", 0))
        subs = int(data.get("subs", 0))
        engagement = float(data.get("engagement", 0))

        if revenue_model:
            prediction = revenue_model.predict(np.array([[earnings, subs, engagement]]))
            predicted_value = float(prediction[0][0])
        else:
            predicted_value = earnings * 1.05  # Fallback Calculation

        return jsonify({"predicted_revenue": predicted_value})
    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        return jsonify({"error": "Invalid request data"})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get port from environment
    app.run(debug=True, host="0.0.0.0", port=port)
