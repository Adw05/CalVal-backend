import pandas as pd
import torchvision.models as models
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model, scaler, encoder, and dataset
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Fallback for LightGBM compatibility issues
    import lightgbm as lgb
    # Try to load the model in a different way
    with open('model.pkl', 'rb') as f:
        model_dict = pickle.load(f)
        if hasattr(model_dict, '_Booster'):
            model = model_dict
        else:
            # If not a direct LightGBM model, try to reconstruct it
            model = lgb.Booster(model_str=model_dict.model_str) if hasattr(model_dict, 'model_str') else model_dict

scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
data = pd.read_csv('data.csv.gz')  

# Define class labels WITHOUT 'Toyota_' prefix to avoid confusion
model_classes = [
    '2000GT', '4Runner', '86', 'Alphard', 'Avalon',
    'Avanza', 'Belta', 'CHR', 'Camry', 'Century',
    'Coaster', 'Corolla', 'Corolla Cross', 'Cressida',
    'Crown', 'FJ Cruiser', 'Fortuner', 'Frontlander',
    'Grand Highlander', 'Granvia', 'Hiace', 'Highlander',
    'Hilux', 'IQ', 'Innova', 'Land Cruiser',
    'Land Cruiser 70', 'Land Cruiser Hard Top', 'Land Cruiser Pick Up',
    'Levin', 'Lite Ace', 'Prado', 'Previa', 'Prius',
    'RAV4', 'Raize', 'Rumion', 'Rush', 'Sequoia',
    'Sienna', 'Starlet', 'Supra', 'Tacoma', 'Tundra',
    'Urban Cruiser', 'Vanguard', 'Veloz', 'Venza',
    'Wigo', 'Wildlander', 'Yaris', 'bZ3', 'bZ4X'
]

# Load the pretrained ResNet50 model architecture
image_model = models.resnet50(pretrained=False)
num_features = image_model.fc.in_features
image_model.fc = torch.nn.Linear(num_features, len(model_classes))

# Load the state dict (model weights)
image_model.load_state_dict(torch.load('toyota_model_V1.pth', map_location=torch.device('cpu')))
image_model.eval()

# Define image transformations (for ResNet50)
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Helper: Tabular price prediction ---
def predict_price_range(model, scaler, encoder, data, model_name, year, mileage, future_year=None):
    model_data = data[data['model'] == model_name].copy()
    if model_data.empty:
        return None, None, None, f"No data found for model: {model_name}"

    # Add/update fields
    model_data['year'] = year
    model_data['age'] = 2025 - year
    model_data['mileage'] = mileage
    model_data['mileage_per_year'] = mileage / (model_data['age'] + 1e-3)
    model_data['is_luxury'] = model_data['model'].str.contains('Cruiser|Prado|Fortuner').astype(int)
    model_data['transmission_code'] = model_data['transmission'].map({'Automatic': 1, 'Manual': 0}).fillna(0)
    model_data['fuel_type'] = model_data['fuel'].map({'Petrol': 1, 'Diesel': 2, 'Hybrid': 3, 'Electric': 4, 'Other': 0}).fillna(0)

    features = [
        'model', 'year', 'mileage', 'age', 'mileage_per_year', 'is_luxury',
        'transmission_code', 'fuel_type', 'body', 'seats', 'cylinder'
    ]
    numerical_cols = ['year', 'mileage', 'age', 'mileage_per_year', 'seats', 'cylinder']

    # Encode and scale
    model_data_encoded = encoder.transform(model_data[features])
    model_data_encoded[numerical_cols] = scaler.transform(model_data_encoded[numerical_cols])

    # Predict
    predictions_log = model.predict(model_data_encoded)
    predictions_aed = np.expm1(predictions_log)

    lower_bound = np.percentile(predictions_aed, 5)
    upper_bound = np.percentile(predictions_aed, 95)
    base_price = np.mean(predictions_aed)

    if future_year:
        years_ahead = future_year - 2025
        depreciation_factor = (1 - 0.07) ** years_ahead
        lower_bound *= depreciation_factor
        upper_bound *= depreciation_factor
        base_price *= depreciation_factor

    return lower_bound, base_price, upper_bound, None

# --- Helper: Predict class from image ---
def predict_model_from_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_tensor = image_transforms(image).unsqueeze(0)

        with torch.no_grad():
            outputs = image_model(image_tensor)
            _, predicted_idx = torch.max(outputs, 1)

        predicted_class_index = predicted_idx.item()
        predicted_class = model_classes[predicted_class_index]
        return predicted_class
    except Exception as e:
        print(f"Error in image prediction: {str(e)}")
        return None

# --- API Routes ---

@app.route('/models', methods=['GET'])
def get_models():
    models_list = sorted(data['model'].unique().tolist())
    return jsonify(models_list)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_input = request.get_json()
        model_name = data_input['model']
        year = int(data_input['year'])
        mileage = int(data_input['mileage'])
        future_year = int(data_input['future_year'])

        lower, base, upper, error = predict_price_range(model, scaler, encoder, data, model_name, year, mileage, future_year)
        if error:
            return jsonify({'error': error}), 400

        return jsonify({
            'model': model_name,
            'year': year,
            'future_year': future_year,
            'lower_bound': int(lower),
            'upper_bound': int(upper)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files and 'image' not in request.form:
            return jsonify({'error': 'No image provided'}), 400

        # Get image bytes
        if 'image' in request.files:
            image_bytes = request.files['image'].read()
        else:
            image_data = request.form['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)

        year = int(request.form.get('year', 2020))
        mileage = int(request.form.get('mileage', 50000))
        future_year = int(request.form.get('future_year', 2025))

        predicted_model = predict_model_from_image(image_bytes)
        if not predicted_model:
            return jsonify({'error': 'Could not identify Toyota model from image'}), 400

        # Use the predicted_model as is because model_classes have no 'Toyota_' prefix
        lower, base, upper, error = predict_price_range(model, scaler, encoder, data, predicted_model, year, mileage, future_year)
        if error:
            return jsonify({'error': error}), 400

        return jsonify({
            'model': predicted_model,
            'year': year,
            'future_year': future_year,
            'lower_bound': int(lower),
            'upper_bound': int(upper)
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 400

# Run app
if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
