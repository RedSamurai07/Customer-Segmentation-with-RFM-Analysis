from flask import Flask, request, jsonify
import pandas as pd
import mlflow.pyfunc
import os

app = Flask(__name__)

MODEL_PATH = "mlruns/0/" 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
     
        return jsonify({
            "status": "success",
            "message": "Customer Segmentation API is active. Send data for RFM scoring.",
            "received_data_samples": len(df)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
