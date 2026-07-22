import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static')
base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, 'handwritten_model.pkl')
if not os.path.exists(model_path):
    from train_model import train_character_recognition
    train_character_recognition()

model = joblib.load(model_path)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Pixels can be passed as an 8x8 array or 64 values representing grayscale (0-16)
        pixels = data.get('pixels')
        if not pixels or len(pixels) != 64:
            return jsonify({'status': 'error', 'message': 'Invalid pixel matrix. Expected 64 features (8x8 grid)'}), 400
            
        pixel_array = np.array(pixels, dtype=np.float32).reshape(1, -1)
        
        # Scale to 0-1 range as done in training
        pixel_scaled = pixel_array / 16.0
        
        probabilities = model.predict_proba(pixel_scaled)[0]
        prediction = int(np.argmax(probabilities))
        
        confidence = float(probabilities[prediction]) * 100.0
        
        prob_dict = {str(digit): round(float(prob) * 100, 2) for digit, prob in enumerate(probabilities)}
        
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'probabilities': prob_dict
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
