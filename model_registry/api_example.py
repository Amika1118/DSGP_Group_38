
# API ENDPOINT EXAMPLE (Flask)

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load predictor
predictor = joblib.load('model_registry/retail_price_predictor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.get_json()
        
        # Convert to DataFrame
        input_df = pd.DataFrame(data['features'])
        
        # Ensure correct feature order
        input_df = input_df[predictor.config['selected_features']]
        
        # Make prediction
        result = predictor.predict_with_confidence(input_df)
        
        # Prepare response
        response = {
            'predictions': result['predictions'].tolist(),
            'confidence_intervals': {
                'upper': result['upper'].tolist(),
                'lower': result['lower'].tolist()
            },
            'metadata': {
                'model_version': '1.0',
                'ensemble_type': predictor.config['ensemble_type'],
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
